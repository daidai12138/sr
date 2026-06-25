# train_sp_1p_3c_sweep.py
# Loss-weight sweep for:
#   loss = w_data*loss_data + w_shape*loss_shape + w_var*loss_var + w_tv*loss_tv + w_mean*loss_mean
#
# Notes:
# - Uses CustomModel from hex.hex_architecture_1p_3c
# - Uses PatchDataset/seed_torch/print_network from hex.utils_point_1p
# - Designed to be a drop-in replacement for train_sp_1p_3c.py with added sweep support.

import os
from os.path import join
import argparse
import math
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# project imports
from hex.hex_architecture_1p_3c import CustomModel
from hex.utils_point_1p import PatchDataset, seed_torch, print_network

from itertools import product
import gc

# ---------------------------
# LR schedule: warmup + cosine
# ---------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------
# Trainable stages
# ---------------------------
def set_trainable_for_stage(model: torch.nn.Module, stage: int, unfreeze_last_k: int = 4):
    """
    stage 0: only heads (+ optional encoder LN)
    stage 1: unfreeze last K encoder layers + LN + heads
    """
    for p in model.parameters():
        p.requires_grad = False

    # heads always train
    for name in ("token_to_grid", "out_head", "out_head_2", "film", "pro_head"):
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True

    if stage == 0:
        # open encoder layer_norm for stability
        if hasattr(model, "visual") and hasattr(model.visual, "beit3"):
            enc = model.visual.beit3.encoder
            if hasattr(enc, "layer_norm"):
                for p in enc.layer_norm.parameters():
                    p.requires_grad = True
        return

    # stage 1
    if hasattr(model, "visual") and hasattr(model.visual, "beit3"):
        enc = model.visual.beit3.encoder
        if hasattr(enc, "layers"):
            for layer in enc.layers[-unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(enc, "layer_norm"):
            for p in enc.layer_norm.parameters():
                p.requires_grad = True


def build_param_groups(model, base_lr_backbone: float, base_lr_head: float, weight_decay: float):
    """token_to_grid/out_head/film use head lr; backbone uses backbone lr; LN/bias no weight decay."""
    def is_no_wd(name: str, param: torch.nn.Parameter) -> bool:
        if param.ndim == 1:
            return True
        if name.endswith(".bias"):
            return True
        return False

    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_head = any(k in n for k in ("token_to_grid", "out_head", "out_head_2", "film", "pro_head"))
        if in_head:
            (head_no_decay if is_no_wd(n, p) else head_decay).append(p)
        else:
            (backbone_no_decay if is_no_wd(n, p) else backbone_decay).append(p)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": base_lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": base_lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"})
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr_head, "weight_decay": weight_decay, "name": "head_decay"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": base_lr_head, "weight_decay": 0.0, "name": "head_no_decay"})
    return groups


# ---------------------------
# Pearson (per-sample, per-channel) in float32
# ---------------------------
def pearson_r_torch(x, y, eps=1e-6):
    # x,y: (B,10,10,C)
    x = x.float()
    y = y.float()
    b, _, _, c = x.shape
    x = x.permute(0, 3, 1, 2).reshape(b, c, -1)  # (B,C,100)
    y = y.permute(0, 3, 1, 2).reshape(b, c, -1)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    cov = (x * y).mean(dim=-1)
    varx = (x * x).mean(dim=-1).clamp_min(eps)
    vary = (y * y).mean(dim=-1).clamp_min(eps)
    r = cov / (varx.sqrt() * vary.sqrt()).clamp_min(eps)
    return r  # (B,C)


def tv_loss(x):
    # x: (B,H,W,C)
    dh = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
    dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dh + dw


def build_csv(img_dir: str, sample_list):
    all_csvs = []
    for sample_id in sample_list:
        folder = join(img_dir, sample_id)
        path_list = os.listdir(folder)
        # assume name ..._<index>.png/jpg
        path_list = sorted(path_list, key=lambda s: int(os.path.splitext(s)[0].rsplit("_", 1)[1]))
        df = pd.DataFrame()
        df["images"] = [join(folder, s) for s in path_list]
        df["sample_id"] = sample_id
        df["img_index"] = [int(os.path.splitext(s)[0].rsplit("_", 1)[1]) for s in path_list]
        all_csvs.append(df)
    return pd.concat(all_csvs).reset_index(drop=True)


def iter_weight_settings(grid: dict):
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def train_one_run(
    args,
    device,
    train_loader,
    val_loader,
    molan_n: int,
    weights: dict,
    run_id: int,
):
    # set per-run seed for model init
    seed_torch(args.seed + run_id)

    # model
    model = CustomModel(
        ckpt_path=args.ckpt_path,
        model_config=args.model_config,
        vocab_size=args.vocab_size,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        molan_n=molan_n,
    ).to(device)

    try:
        print_network(model)
    except Exception:
        pass

    # trainable stage
    set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)

    # optimizer / scheduler
    param_groups = build_param_groups(
        model,
        base_lr_backbone=args.lr_backbone,
        base_lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # dirs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{ts}_run{run_id}"
        f"_wd{weights['w_data']}_ws{weights['w_shape']}_wv{weights['w_var']}"
        f"_wtv{weights['w_tv']}_wm{weights['w_mean']}"
    )
    writer_dir = join(args.save_dir, "runs", run_name)
    ckpt_dir = join(args.save_dir, "checkpoints", run_name)
    os.makedirs(writer_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)

    # write config
    writer.add_text("loss_weights", str(weights), global_step=0)
    writer.add_text("args", str(vars(args)), global_step=0)

    # optional graph (fixed correct input shapes)
    try:
        init_img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
        init_point = torch.zeros((1, 1, molan_n), device=device)
        writer.add_graph(model, (init_img, init_point))
    except Exception:
        pass

    best_val_mse = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        # switch stage after warmup
        if epoch == args.warmup_epochs:
            set_trainable_for_stage(model, stage=1, unfreeze_last_k=args.unfreeze_last_k)
            # IMPORTANT: rebuild optimizer param groups to include newly trainable params
            param_groups = build_param_groups(
                model,
                base_lr_backbone=args.lr_backbone,
                base_lr_head=args.lr_head,
                weight_decay=args.weight_decay,
            )
            optimizer = torch.optim.AdamW(param_groups)
            total_steps = args.epochs * max(1, len(train_loader))
            warmup_steps = args.warmup_epochs * max(1, len(train_loader))
            scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
            scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        model.train()

        sum_loss = 0.0
        sum_mse = 0.0
        sum_mae = 0.0
        sum_r, cnt_r = 0.0, 0

        loss_data_sum = loss_shape_sum = loss_var_sum = loss_tv_sum = loss_mean_sum = 0.0
        loss_steps = 0

        train_p = tqdm.tqdm(train_loader, desc=f"[RUN {run_id}] Train {epoch+1}/{args.epochs}", leave=False)
        for batch in train_p:
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            points = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs, points)

            # ---- loss computed in float32 for stability ----
            pred = outputs.float()
            gt = labels.float()

            loss_data = F.smooth_l1_loss(pred, gt, beta=0.1)

            pred_dm = pred - pred.mean(dim=(1, 2), keepdim=True)
            gt_dm = gt - gt.mean(dim=(1, 2), keepdim=True)
            loss_shape = ((pred_dm - gt_dm) ** 2).mean()

            loss_mean = pred.mean(dim=(1, 2)).abs().mean()

            pred_std = pred.std(dim=(1, 2), unbiased=False)
            gt_std = gt.std(dim=(1, 2), unbiased=False)
            loss_var = (pred_std - gt_std).abs().mean()

            loss_tv = tv_loss(pred)

            loss = (
                float(weights["w_data"]) * loss_data
                + float(weights["w_shape"]) * loss_shape
                + float(weights["w_var"]) * loss_var
                + float(weights["w_tv"]) * loss_tv
                + float(weights["w_mean"]) * loss_mean
            )

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # metrics
            with torch.no_grad():
                mse = F.mse_loss(pred, gt).item()
                mae = F.l1_loss(pred, gt).item()
                r = pearson_r_torch(pred, gt).detach()
                sum_r += r.sum().item()
                cnt_r += r.numel()

            sum_loss += loss.item()
            sum_mse += mse
            sum_mae += mae

            loss_data_sum += loss_data.item()
            loss_shape_sum += loss_shape.item()
            loss_var_sum += loss_var.item()
            loss_tv_sum += loss_tv.item()
            loss_mean_sum += loss_mean.item()
            loss_steps += 1

            train_p.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = sum_loss / max(1, len(train_loader))
        train_mse = sum_mse / max(1, len(train_loader))
        train_mae = sum_mae / max(1, len(train_loader))
        train_pearson = (sum_r / max(1, cnt_r)) if cnt_r > 0 else 0.0

        if loss_steps > 0:
            print(
                f"[Epoch {epoch+1}] "
                f"loss_data={loss_data_sum/loss_steps:.6f} | "
                f"loss_shape={loss_shape_sum/loss_steps:.6f} | "
                f"loss_var={loss_var_sum/loss_steps:.6f} | "
                f"loss_tv={loss_tv_sum/loss_steps:.6f} | "
                f"loss_mean={loss_mean_sum/loss_steps:.6f}"
            )
        print(f"  train: loss={train_loss:.6f} mse={train_mse:.6f} mae={train_mae:.6f}, pearson={train_pearson:.4f}")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("train/mae", train_mae, epoch)
        writer.add_scalar("train/pearson", train_pearson, epoch)

        # ---- val ----
        model.eval()
        v_mse_sum = 0.0
        v_mae_sum = 0.0
        v_sum_r, v_cnt_r = 0.0, 0

        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"[RUN {run_id}] Val {epoch+1}/{args.epochs}", leave=False):
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                points = batch[2].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(inputs, points)

                pred = outputs.float()
                gt = labels.float()

                v_mse_sum += F.mse_loss(pred, gt).item()
                v_mae_sum += F.l1_loss(pred, gt).item()
                r = pearson_r_torch(pred, gt).detach()
                v_sum_r += r.sum().item()
                v_cnt_r += r.numel()

        val_mse = v_mse_sum / max(1, len(val_loader))
        val_rmse = math.sqrt(val_mse)
        val_mae = v_mae_sum / max(1, len(val_loader))
        val_pearson = (v_sum_r / max(1, v_cnt_r)) if v_cnt_r > 0 else 0.0

        print(f"  val  : mse={val_mse:.6f} rmse={val_rmse:.6f} mae={val_mae:.6f}, pearson={val_pearson:.4f}")

        writer.add_scalar("val/mse", val_mse, epoch)
        writer.add_scalar("val/rmse", val_rmse, epoch)
        writer.add_scalar("val/mae", val_mae, epoch)
        writer.add_scalar("val/pearson", val_pearson, epoch)

        # save ckpt (best by val_mse)
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse

        def trainable_state_dict(m):
            return {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}

        ckpt = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_val_mse": best_val_mse,
            "ft_model": trainable_state_dict(model),
            "loss_weights": weights,
            "args": vars(args),
        }
        torch.save(ckpt, join(ckpt_dir, "checkpoint_last_ft.pth"))
        if is_best:
            torch.save(ckpt, join(ckpt_dir, "checkpoint_best_ft.pth"))

    writer.close()
    print(f"[RUN {run_id}] Finished. best_val_mse={best_val_mse:.6f}")

    # free memory between runs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--molan-n", type=int, default=3)

    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--img-dir", type=str, default="./hex/sample_data/HE")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")
    parser.add_argument("--train-list", default=["N2", "N3", "N4", "N5", "P1", "P2", "P3", "P4"])
    parser.add_argument("--val-list",  default=["N1", "P5"])
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)

    parser.add_argument("--subsample-frac", type=float, default=0.5, help="subsample train/val csvs by this fraction")

    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-last-k", type=int, default=2)

    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # model args
    parser.add_argument("--ckpt-path", type=str, default="./MUSK/model.safetensors")
    parser.add_argument("--model-config", type=str, default="musk_large_patch16_384")
    parser.add_argument("--vocab-size", type=int, default=64010)
    parser.add_argument("--grid-h", type=int, default=10)
    parser.add_argument("--grid-w", type=int, default=10)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # single-run loss weights (default matches your formula)
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-shape", type=float, default=1.0)
    parser.add_argument("--w-var", type=float, default=0.5)
    parser.add_argument("--w-tv", type=float, default=0.002)
    parser.add_argument("--w-mean", type=float, default=0.1)

    # sweep
    parser.add_argument("--sweep", action="store_true", help="sweep loss weights in one run")
    parser.add_argument("--sweep-max-runs", type=int, default=999999, help="cap number of sweep runs")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load molan
    molan_path = join(args.csv_dir, f"molan_{args.molan_n}_1.csv")
    molan = pd.read_csv(molan_path, index_col=0)
    molan_n = int(molan.shape[0])
    if molan_n != args.molan_n:
        print(f"[WARN] molan rows={molan_n} != molan_n={args.molan_n}, using molan rows.")
        args.molan_n = molan_n

    # build csvs
    train_csvs = build_csv(args.img_dir, args.train_list)
    val_csvs = build_csv(args.img_dir, args.val_list)

    if args.subsample_frac < 1.0:
        train_csvs = train_csvs.sample(frac=args.subsample_frac, random_state=args.seed).reset_index(drop=True)
        val_csvs = val_csvs.sample(frac=args.subsample_frac, random_state=args.seed).reset_index(drop=True)

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    train_dataset = PatchDataset(train_csvs, args.data_dir, molan, args.train_list, transform_train)
    val_dataset = PatchDataset(val_csvs, args.data_dir, molan, args.val_list, transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    print("train size:", len(train_dataset), "val size:", len(val_dataset))
    print("molan_n:", molan_n, "img_size:", args.img_size)

    # loss weights settings
    default_w = dict(
        w_data=float(args.w_data),
        w_shape=float(args.w_shape),
        w_var=float(args.w_var),
        w_tv=float(args.w_tv),
        w_mean=float(args.w_mean),
    )

    # edit this grid as you like
    sweep_grid = dict(
        w_data=[1.0],
        w_shape=[0.5, 1.0, 2.0],
        w_var=[0.1, 0.3, 0.5, 1.0],
        w_tv=[0.0, 0.001, 0.002, 0.005],
        w_mean=[0.0, 0.05, 0.1, 0.2],
    )

    weight_settings = [default_w]
    if args.sweep:
        weight_settings = list(iter_weight_settings(sweep_grid))[: args.sweep_max_runs]

    print(f"[SWEEP] total runs = {len(weight_settings)}")
    for rid, w in enumerate(weight_settings):
        print(f"\n===== RUN {rid}/{len(weight_settings)-1} weights={w} =====")
        train_one_run(
            args=args,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            molan_n=molan_n,
            weights=w,
            run_id=rid,
        )

    print("All runs finished.")


def train_one_run(args, device, train_loader, val_loader, molan, run_tag,
                  w_data=1.0, w_shape=1.0, w_var=0.5, w_tv=0.002, w_mean=0.1):
    """
    单次训练：给定一组 loss 权重，跑完整 epochs，并把 ckpt/tensorboard 写到独立目录
    返回：best_val_mse, best_val_pearson
    """
    # ---- 重要：每个 run 重新设随机种子，保证可复现对比 ----
    try:
        seed_torch(args.seed)
    except Exception:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ========== Model ==========
    model = CustomModel(
        ckpt_path=args.ckpt_path,
        model_config=args.model_config,
        vocab_size=args.vocab_size,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        molan_n=args.molan_n,
    ).to(device)

    # robust loss（你虽然现在没用robust，但先保留不影响）
    criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=args.molan_n,
        float_dtype=torch.float32,
        device=device,
    )

    start_epoch = 0
    best_val_mse = float("inf")
    best_val_pearson = -1e9
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ========== 日志/ckpt 目录（每组权重独立） ==========
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + run_tag
    writer_dir = join(args.save_dir, "runs", run_name)
    ckpt_dir = join(args.save_dir, "checkpoints", run_name)
    os.makedirs(writer_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)

    # 画图（可选）
    try:
        init_img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
        init_point = torch.zeros((1, 1, args.molan_n), device=device)
        writer.add_graph(model, (init_img, init_point))
    except Exception:
        pass

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    # stage0
    set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)

    param_groups = build_param_groups(
        model,
        base_lr_backbone=args.lr_backbone,
        base_lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    optimizer.add_param_group({
        "params": list(criterion_ad.parameters()),
        "lr": args.lr_robust,
        "weight_decay": 0.0,
        "name": "robust_params"
    })
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.warmup_epochs:
            # 你原代码这里写的是 stage=0（保持一致）；如果你想解冻请改成 stage=1
            set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)
            param_groups = build_param_groups(
                model,
                base_lr_backbone=args.lr_backbone,
                base_lr_head=args.lr_head,
                weight_decay=args.weight_decay,
            )
            optimizer = torch.optim.AdamW(param_groups)
            optimizer.add_param_group({
                "params": list(criterion_ad.parameters()),
                "lr": args.lr_robust,
                "weight_decay": 0.0,
                "name": "robust_params"
            })
            scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0
        sum_r, cnt = 0.0, 0

        train_p = tqdm.tqdm(train_loader, desc=f"[{run_tag}] Train {epoch+1}/{args.epochs}", leave=False)
        for batch in train_p:
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            points = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs, points)  # (B,10,10,C)

            pred = outputs
            gt = labels

            # 1) 数据项
            loss_data = torch.nn.functional.smooth_l1_loss(pred, gt, beta=0.1)

            # 2) 去均值形状项
            pred_dm = pred - pred.mean(dim=(1,2), keepdim=True)
            gt_dm   = gt   - gt.mean(dim=(1,2), keepdim=True)
            loss_shape = ((pred_dm - gt_dm)**2).mean()

            # 3) mean项
            loss_mean = pred.mean(dim=(1,2)).abs().mean()

            # 4) 方差项
            pred_std = pred.std(dim=(1,2))
            gt_std   = gt.std(dim=(1,2))
            loss_var = (pred_std - gt_std).abs().mean()

            # 5) TV项
            def tv_loss(x):
                dh = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
                dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
                return dh + dw
            loss_tv = tv_loss(pred)

            # ✅ 用 sweep 的权重来组合 loss（替换你原来的固定写法）
            loss = (w_data * loss_data
                    + w_shape * loss_shape
                    + w_var * loss_var
                    + w_tv * loss_tv
                    + w_mean * loss_mean)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=args.grad_clip
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                diff = (outputs.to(torch.float32) - labels.to(torch.float32))
                train_mse_sum += float((diff * diff).sum().item())
                train_mae_sum += float(diff.abs().sum().item())
                train_count += int(diff.numel())
                train_loss_sum += float(loss.item())

                r = pearson_r_torch(outputs, labels, eps=1e-6)
                sum_r += r.sum().item()
                cnt += r.numel()

            train_p.set_postfix(loss=f"{loss.item():.4f}", pearson=f"{(sum_r/max(1,cnt)):.4f}")

        train_loss = train_loss_sum / max(1, len(train_loader))
        train_mse = train_mse_sum / max(1, train_count)
        train_mae = train_mae_sum / max(1, train_count)
        train_pearson = sum_r / max(1, cnt)

        # ---- val ----
        model.eval()
        val_mse_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0
        v_sum_r, v_cnt = 0.0, 0

        with torch.no_grad():
            val_p = tqdm.tqdm(val_loader, desc=f"[{run_tag}] Val {epoch+1}/{args.epochs}", leave=False)
            for batch in val_p:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                points = batch[2].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(inputs, points)

                diff = (outputs.to(torch.float32) - labels.to(torch.float32))
                val_mse_sum += float((diff * diff).sum().item())
                val_mae_sum += float(diff.abs().sum().item())
                val_count += int(diff.numel())

                r = pearson_r_torch(outputs, labels, eps=1e-6)
                v_sum_r += r.sum().item()
                v_cnt += r.numel()

        val_mse = val_mse_sum / max(1, val_count)
        val_mae = val_mae_sum / max(1, val_count)
        val_pearson = v_sum_r / max(1, v_cnt)

        # 记录 best（你也可以改成按 pearson 选）
        if val_mse < best_val_mse:
            best_val_mse = val_mse
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson

        # tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Metrics/train_mse", train_mse, epoch + 1)
        writer.add_scalar("Metrics/train_mae", train_mae, epoch + 1)
        writer.add_scalar("Metrics/train_pearson", train_pearson, epoch + 1)
        writer.add_scalar("Metrics/val_mse", val_mse, epoch + 1)
        writer.add_scalar("Metrics/val_mae", val_mae, epoch + 1)
        writer.add_scalar("Metrics/val_pearson", val_pearson, epoch + 1)

        print(f"\n[{run_tag}] Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_p={train_pearson:.4f} | "
              f"val_mse={val_mse:.4f} val_p={val_pearson:.4f}")

        # 保存 last + best（按 val_mse）
        ckpt = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_val_mse": best_val_mse,
            "args": vars(args),
            "loss_weights": {
                "w_data": w_data, "w_shape": w_shape, "w_var": w_var, "w_tv": w_tv, "w_mean": w_mean
            },
            "model": finetuned_state_dict(model),
        }
        torch.save(ckpt, join(ckpt_dir, "checkpoint_last.pth"))
        if val_mse <= best_val_mse + 1e-12:
            torch.save(ckpt, join(ckpt_dir, "checkpoint_best.pth"))

    writer.close()

    # 释放显存
    del model, optimizer, scheduler, criterion_ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_mse, best_val_pearson




if __name__ == "__main__":
    main()
