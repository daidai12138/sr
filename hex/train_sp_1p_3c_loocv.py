# train_hex_token_grid.py
# 适配：模型输出 (B, 10, 10, molan_n)，默认 molan_n=500
# 训练更强：分组学习率（backbone vs token_to_grid vs robust loss params） + warmup+cosine + grad clip + best ckpt
#
# 运行示例：
#   python train_hex_token_grid.py
#   python train_hex_token_grid.py --molan-n 500 --epochs 30 --batch-size 32
#
# 依赖：
#   - 你的数据与 PatchDataset（来自 hex.utils）
#   - 你的模型 CustomModel（来自 hex.hex_architecture_3），其 forward(img) 返回 (B,10,10,molan_n)
#tensorboard --logdir ./experiment/runs --port 6006

import os
from os.path import join
import argparse
import time
import math
import numpy as np
import pandas as pd
import tqdm
import os

from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import robust_loss_pytorch
from scipy.stats import pearsonr

# 你项目里已有的
from hex.hex_architecture_1p_nulear import CustomModel
from hex.utils_point_1p import *  # PatchDataset, seed_torch, print_network 等

# ckpt = torch.load("finetuned_best.pth", map_location="cpu")
# model = CustomModel(...).to(device)  # 这里会按你原逻辑加载 MUSK base ckpt :contentReference[oaicite:4]{index=4}
# model.load_state_dict(ckpt["model_finetuned"], strict=False)
# ---------------------------
# 指标：Streaming Pearson（按通道）
# ---------------------------


# ---------------------------
# LR schedule: warmup + cosine
# ---------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine from 1 -> 0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def finetuned_state_dict(model: torch.nn.Module):
    # 记录所有需要训练的“模块前缀”，这样能把同模块的 buffer 也一起带上
    prefixes = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            prefixes.add(n.rsplit(".", 1)[0])  # 去掉最后的 weight/bias 名

    def keep(k: str) -> bool:
        return any(k == pref or k.startswith(pref + ".") for pref in prefixes)

    return {k: v.cpu() for k, v in model.state_dict().items() if keep(k)}


# ---------------------------
# 参数分组：backbone vs token_to_grid vs robust loss params
# ---------------------------
def set_trainable_for_stage(model: torch.nn.Module, stage: int, unfreeze_last_k: int = 2):
    """
    stage 0: 只训练 token_to_grid（以及可选 layer_norm）
    stage 1: 解冻最后 K 层 encoder + layer_norm + token_to_grid
    """
    # 先全冻
    for p in model.parameters():
        p.requires_grad = False

    # token_to_grid 永远训练
    if hasattr(model, "token_to_grid"):
        for p in model.token_to_grid.parameters():
            p.requires_grad = True
    if hasattr(model, "out_head"):
        for p in model.out_head.parameters():
            p.requires_grad = True
    if hasattr(model, "out_head_2"):
        for p in model.out_head_2.parameters():
            p.requires_grad = True
    if hasattr(model, "film"):
        for p in model.film.parameters():
            p.requires_grad = True

    # stage 0: 可以选择放开 layer_norm（通常更稳）
    if stage == 0:
        if hasattr(model, "visual") and hasattr(model.visual, "beit3"):
            if hasattr(model.visual.beit3, "encoder") and hasattr(model.visual.beit3.encoder, "layer_norm"):
                for p in model.visual.beit3.encoder.layer_norm.parameters():
                    p.requires_grad = True   
        return

    # stage 1: 解冻最后 K 层
    if hasattr(model, "visual") and hasattr(model.visual, "beit3"):
        enc = model.visual.beit3.encoder
        # 最后 K 层
        if hasattr(enc, "layers"):
            for layer in enc.layers[-unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # encoder layer_norm
        if hasattr(enc, "layer_norm"):
            for p in enc.layer_norm.parameters():
                p.requires_grad = True


def build_param_groups(model, base_lr_backbone: float, base_lr_head: float, weight_decay: float):
    """
    经验：token_to_grid 用更大学习率；backbone 用更小学习率；LN/bias 不做 weight_decay
    """
    def is_no_wd(name: str, param: torch.nn.Parameter):
        if param.ndim == 1:
            return True  # LayerNorm/BatchNorm 权重通常 1D
        if name.endswith(".bias"):
            return True
        return False

    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        in_head = any(k in n for k in ("token_to_grid", "out_head", "out_head_2","film"))
        if in_head:
            (head_no_decay if is_no_wd(n, p) else head_decay).append(p)
        else:
            (backbone_no_decay if is_no_wd(n, p) else backbone_decay).append(p)

    param_groups = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": base_lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"})
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": base_lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": base_lr_head, "weight_decay": weight_decay, "name": "head_decay"})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": base_lr_head, "weight_decay": 0.0, "name": "head_no_decay"})

    return param_groups

import torch

def pearson_r_torch(x, y, eps=1e-6):
    # x,y: (b,10,10,c)
    b, _, _, c = x.shape
    x = x.permute(0, 3, 1, 2).reshape(b, c, -1)  # (b,c,100)
    y = y.permute(0, 3, 1, 2).reshape(b, c, -1)

    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    cov = (x * y).mean(dim=-1)
    stdx = torch.sqrt((x * x).mean(dim=-1))
    stdy = torch.sqrt((y * y).mean(dim=-1))

    r = cov / (stdx * stdy + eps)   # (b,c)
    return r

    # ---------------------------
    # 数据准备工具
    # ---------------------------

# ---------------------------
# 主训练
# ---------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--molan-n", type=int, default=3)

    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--img-dir", type=str, default="./hex/sample_data/HE")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")

    # 留一交叉验证（LOOCV）
    # 默认会对下面 10 个样本做 LOOCV：每次留 1 个做验证，其余 9 个做训练。
    # 你也可以用 --holdout N3 只跑某一个 fold（验证集=指定样本）。
    parser.add_argument(
        "--all-list",
        nargs="+",
        default=["N1", "N2", "N3", "N4", "N5", "P1", "P2", "P3", "P4", "P5"],
        help="all sample ids used for LOOCV",
    )
    parser.add_argument(
        "--loocv",
        action=getattr(argparse, "BooleanOptionalAction", "store_true"),
        default=True,
        help="use leave-one-out cross validation (default: True)",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default="",
        help="if set (e.g. N3), only run the fold where validation is this sample id",
    )

    # 兼容旧用法：当 --no-loocv 时，使用 train/val list
    parser.add_argument("--train-list", nargs="+", default=["N2", "N3", "N4", "N5", "P1", "P2", "P4", "P5"])
    parser.add_argument("--val-list", nargs="+", default=["N1", "P3"])
    parser.add_argument("--img-size", type=int, default=384)                    
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)

    
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-last-k", type=int, default=2)

    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-robust", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)  
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--mse-weight", type=float, default=0.1, help="robust loss + mse_weight * MSE")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint .pth (optional)")

    # 模型相关（按你现在的 CustomModel 参数）
    parser.add_argument("--ckpt-path", type=str, default="./MUSK/model.safetensors")
    parser.add_argument("--model-config", type=str, default="musk_large_patch16_384")
    parser.add_argument("--vocab-size", type=int, default=64010)
    parser.add_argument("--grid-h", type=int, default=10)
    parser.add_argument("--grid-w", type=int, default=10)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pearson-weight", type=float, default=0.1)

    args = parser.parse_args()

    # 基础设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    try:
        seed_torch(args.seed)
    except Exception:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # 读 molan
    molan_path = join(args.csv_dir, f"molan_{args.molan_n}_1.csv")
    molan = pd.read_csv(molan_path, index_col=0)
    num_outputs = molan.shape[0]
    if num_outputs != args.molan_n:
        print(f"[WARN] molan rows={num_outputs} != molan_n={args.molan_n}，以 molan rows 为准。")
        args.molan_n = num_outputs
    
    
    def build_csv(sample_list):
        all_csvs = []
        for sample_id in sample_list:
            path_list = os.listdir(join(args.img_dir, sample_id))
            path_list = sorted(path_list, key=lambda s: int(os.path.splitext(s)[0].rsplit("_", 1)[1]))
            df = pd.DataFrame()
            df["images"] = [join(args.img_dir, sample_id, s) for s in path_list]
            df["sample_id"] = sample_id
            df["img_index"] = [int(os.path.splitext(s)[0].rsplit("_", 1)[1]) for s in path_list]
            all_csvs.append(df)
        return pd.concat(all_csvs).reset_index(drop=True)    


    # transforms（更强：轻量增强）
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    # ---------------------------
    # 单个 fold 的训练
    # ---------------------------
    def run_one_fold(train_list, val_list, fold_tag: str, fold_seed: int):
        """训练一个 fold，并返回该 fold 的最佳验证集 MSE 等信息。"""

        # 每个 fold 单独 seed，保证可复现
        try:
            seed_torch(fold_seed)
        except Exception:
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)

        # 构建 train/val csv 列表
        train_csvs = build_csv(train_list)
        val_csvs = build_csv(val_list)

        # 可选：子采样（保持你原来的逻辑）
        train_csvs = train_csvs.sample(frac=0.5, random_state=fold_seed).reset_index(drop=True)
        val_csvs = val_csvs.sample(frac=0.5, random_state=fold_seed).reset_index(drop=True)

        # Dataset
        train_dataset = PatchDataset(train_csvs, args.data_dir, molan, train_list, transform_train)
        val_dataset = PatchDataset(val_csvs, args.data_dir, molan, val_list, transform_val)

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

        print("\n" + "=" * 80)
        print(f"[FOLD] {fold_tag} | train={train_list} | val={val_list}")
        print("train size:", len(train_dataset), "val size:", len(val_dataset))
        print("molan_n:", args.molan_n, "img_size:", args.img_size)

        # Model：输出应为 (B, grid_h, grid_w, molan_n)
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

        try:
            print_network(model)
        except Exception:
            pass

        # robust loss（回归任务很合适）
        criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=args.molan_n,
            float_dtype=torch.float32,
            device=device,
        )

        start_epoch = 0
        best_val_mse = float("inf")
        best_epoch = -1
        global_step = 0
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # 保存目录（每个 fold 独立）
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{fold_tag}"
        writer_dir = join(args.save_dir, "runs", run_name)
        ckpt_dir = join(args.save_dir, "checkpoints", run_name)
        os.makedirs(writer_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir)

        # 你原来的 add_graph（如果 shape 不匹配可忽略）
        try:
            init_img = torch.zeros((1, args.molan_n, args.img_size, args.img_size), device=device)
            init_point = torch.zeros((1, 1, args.molan_n), device=device)
            writer.add_graph(model, (init_img, init_point))
        except Exception as e:
            print(f"[WARN] add_graph failed: {e}")

        # 训练总步数（用于 warmup+cosine）
        total_steps = args.epochs * max(1, len(train_loader))
        warmup_steps = args.warmup_epochs * max(1, len(train_loader))

        # 先设 stage0（只训练 token_to_grid + LN）
        set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)

        # optimizer（param groups）
        param_groups = build_param_groups(
            model,
            base_lr_backbone=args.lr_backbone,
            base_lr_head=args.lr_head,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
        optimizer.add_param_group({"params": list(criterion_ad.parameters()), "lr": args.lr_robust, "weight_decay": 0.0, "name": "robust_params"})

        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

        # resume（包含 optimizer/scheduler/scaler）
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location="cpu")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            else:
                # 兼容：如果你给的是纯 state_dict
                model.load_state_dict(ckpt, strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] is not None and torch.cuda.is_available():
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_mse = float(ckpt.get("best_val_mse", best_val_mse))
            global_step = int(ckpt.get("global_step", 0))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            print(f"[RESUME] from {args.resume} | start_epoch={start_epoch} best_val_mse={best_val_mse:.6f}")

        def save_ckpt(epoch_idx: int, is_best: bool):
            ckpt = {
                "fold": fold_tag,
                "train_list": list(train_list),
                "val_list": list(val_list),
                "epoch": epoch_idx,
                "best_epoch": best_epoch,
                "global_step": global_step,
                "best_val_mse": best_val_mse,
                "model": finetuned_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": (scaler.state_dict() if torch.cuda.is_available() else None),
                "args": vars(args),
            }
            torch.save(ckpt, join(ckpt_dir, "checkpoint_last_ft.pth"))
            if is_best:
                torch.save(ckpt, join(ckpt_dir, "checkpoint_best_ft.pth"))

        # ---------------------------
        # 训练循环
        # ---------------------------
        for epoch in range(start_epoch, args.epochs):
            # stage 切换：warmup 结束后解冻 backbone 最后 K 层
            if epoch == args.warmup_epochs:
                set_trainable_for_stage(model, stage=1, unfreeze_last_k=args.unfreeze_last_k)
                # 重新构建 param groups（因为 requires_grad 变了）
                param_groups = build_param_groups(
                    model,
                    base_lr_backbone=args.lr_backbone,
                    base_lr_head=args.lr_head,
                    weight_decay=args.weight_decay,
                )
                optimizer = torch.optim.AdamW(param_groups)
                optimizer.add_param_group({"params": list(criterion_ad.parameters()), "lr": args.lr_robust, "weight_decay": 0.0, "name": "robust_params"})
                scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
                print(f"[STAGE] epoch={epoch} -> unfreeze last {args.unfreeze_last_k} encoder layers")

            # -------- train --------
            model.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_mae_sum = 0.0
            train_count = 0
            loss_data_sum = 0.0
            loss_corr_sum = 0.0
            loss_var_sum = 0.0
            loss_tv_sum = 0.0
            loss_steps = 0
            sum_r, cnt = 0.0, 0

            train_p = tqdm.tqdm(train_loader, desc=f"Train {fold_tag} {epoch+1}/{args.epochs}", leave=False)
            for batch in train_p:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                points = batch[2].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(inputs, points)  # (B,10,10,C)

                pred = outputs
                gt = labels
                loss_data = torch.nn.functional.smooth_l1_loss(pred, gt, beta=0.1)
                loss_corr = 1.0 - pearson_r_torch(pred, gt, eps=1e-6).mean()
                pred_std = pred.std(dim=(1, 2))
                gt_std = gt.std(dim=(1, 2))
                loss_var = (pred_std - gt_std).abs().mean()

                def tv_loss(x):
                    dh = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
                    dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
                    return dh + dw

                loss_tv = tv_loss(pred)

                # 你原来的 loss 组合
                loss = 1.0 * loss_data

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

                # 统计（按元素平均）
                if (global_step % 10) == 0:
                    with torch.no_grad():
                        out32 = outputs.detach().to(torch.float32)
                        lab32 = labels.detach().to(torch.float32)
                        B, H, W, C = out32.shape
                        elem = float(B * H * W * C)
                        diff = out32 - lab32

                        train_loss_sum += float(loss.detach().item()) * elem
                        train_count += int(elem)
                        train_mse_sum += float((diff * diff).sum().item())
                        train_mae_sum += float(diff.abs().sum().item())

                        train_pearson = pearson_r_torch(out32, lab32, eps=1e-6)
                        sum_r += float(train_pearson.sum().item())
                        cnt += int(train_pearson.numel())

                        loss_data_sum += float(loss_data.detach().item())
                        loss_corr_sum += float(loss_corr.detach().item())
                        loss_var_sum += float(loss_var.detach().item())
                        loss_tv_sum += float(loss_tv.detach().item())
                        loss_steps += 1

            # train epoch metrics
            t_pearson = sum_r / max(1, cnt)
            train_loss = train_loss_sum / max(1, train_count)
            train_mse = train_mse_sum / max(1, train_count)
            train_mae = train_mae_sum / max(1, train_count)
            ld_avg = loss_data_sum / max(1, loss_steps)
            ls_avg = loss_corr_sum / max(1, loss_steps)
            lv_avg = loss_var_sum / max(1, loss_steps)
            ltv_avg = loss_tv_sum / max(1, loss_steps)

            writer.add_scalar("LossComp/train_loss_data", ld_avg, epoch + 1)
            writer.add_scalar("LossComp/train_loss_corr", ls_avg, epoch + 1)
            writer.add_scalar("LossComp/train_loss_var", lv_avg, epoch + 1)
            writer.add_scalar("LossComp/train_loss_tv", ltv_avg, epoch + 1)

            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("Metrics/train_mse", train_mse, epoch + 1)
            writer.add_scalar("Metrics/train_mae", train_mae, epoch + 1)
            writer.add_scalar("Metrics/train_pearson", t_pearson, epoch + 1)

            # -------- val --------
            model.eval()
            val_mse_sum = 0.0
            val_mae_sum = 0.0
            val_count = 0
            v_sum_r, v_cnt = 0.0, 0

            with torch.no_grad():
                val_p = tqdm.tqdm(val_loader, desc=f"Val {fold_tag} {epoch+1}/{args.epochs}", leave=False)
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

                    val_pearson = pearson_r_torch(outputs, labels, eps=1e-6)
                    v_sum_r += float(val_pearson.sum().item())
                    v_cnt += int(val_pearson.numel())

            val_mse = val_mse_sum / max(1, val_count)
            val_mae = val_mae_sum / max(1, val_count)
            val_pearson = v_sum_r / max(1, v_cnt)
            val_rmse = math.sqrt(max(0.0, val_mse))

            writer.add_scalar("Metrics/val_mse", val_mse, epoch + 1)
            writer.add_scalar("Metrics/val_rmse", val_rmse, epoch + 1)
            writer.add_scalar("Metrics/val_mae", val_mae, epoch + 1)
            writer.add_scalar("Metrics/val_pearson", val_pearson, epoch + 1)

            print(f"\n[{fold_tag}] Epoch {epoch+1}/{args.epochs}")
            print(
                f"  loss_data={ld_avg:.6f} | loss_corr={ls_avg:.6f} | loss_var={lv_avg:.6f} | loss_tv={ltv_avg:.6f}"
            )
            print(f"  train: loss={train_loss:.6f} mse={train_mse:.6f} mae={train_mae:.6f} pearson={t_pearson:.4f}")
            print(f"  val  : mse={val_mse:.6f} rmse={val_rmse:.6f} mae={val_mae:.6f} pearson={val_pearson:.4f}")

            # -------- save ckpt --------
            is_best = val_mse < best_val_mse
            if is_best:
                best_val_mse = val_mse
                best_epoch = epoch + 1
            save_ckpt(epoch_idx=epoch + 1, is_best=is_best)

        writer.close()
        print(f"[FOLD DONE] {fold_tag} | best_val_mse={best_val_mse:.6f} @ epoch={best_epoch}")

        return {
            "fold": fold_tag,
            "val_id": val_list[0] if len(val_list) == 1 else "+".join(val_list),
            "best_val_mse": float(best_val_mse),
            "best_epoch": int(best_epoch),
            "run_name": run_name,
        }

    # ---------------------------
    # LOOCV 外层循环
    # ---------------------------
    results = []
    if args.loocv:
        all_ids = list(args.all_list)
        if args.holdout:
            if args.holdout not in all_ids:
                raise ValueError(f"--holdout {args.holdout} not in --all-list {all_ids}")
            holdouts = [args.holdout]
        else:
            holdouts = all_ids
        start_from = 9          
        for i, val_id in enumerate(holdouts):
            if i < start_from:
                continue
            train_list = [s for s in all_ids if s != val_id]
            val_list = [val_id]
            fold_tag = f"LOO_{val_id}"
            results.append(run_one_fold(train_list, val_list, fold_tag=fold_tag, fold_seed=args.seed + i))
    else:
        # 兼容旧用法：手动 train/val split
        results.append(run_one_fold(args.train_list, args.val_list, fold_tag="SPLIT", fold_seed=args.seed))

    # 保存/打印汇总
    df = pd.DataFrame(results)
    summary_path = join(args.save_dir, f"loocv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print("All folds summary:")
    print(df)
    if len(df) > 1:
        print(f"mean(best_val_mse)={df['best_val_mse'].mean():.6f} | std={df['best_val_mse'].std(ddof=1):.6f}")
    print(f"[Saved] {summary_path}")

    print("Finished Training.")


if __name__ == "__main__":
    main()
