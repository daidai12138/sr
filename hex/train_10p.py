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
from hex.hex_architecture_10p import CustomModel
from hex.utils_10p import *  # PatchDataset, seed_torch, print_network 等



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
    if hasattr(model, "regression_head"):
        for p in model.token_to_grid.parameters():
            p.requires_grad = True
    if hasattr(model, "regression_head1"):
        for p in model.out_head.parameters():
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

        in_head = any(k in n for k in ("regression_head", "regression_head1", "out_head_2","film"))
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--molan-n", type=int, default=163c1)

    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--img-dir", type=str, default="./hex/sample_data/he_all")
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

    fold_seed = args.seed

    for sample in args.all_list:
        print("processing sample------------------")
        # 每个 fold 单独 seed，保证可复现
        try:
            seed_torch(fold_seed)
        except Exception:
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)

        all_points = [(i, j) for i in range(10) for j in range(10)]

        train_list = random.sample(all_points, 10)
        val_list = list(set(all_points) - set(train_list))


        # Dataset
        train_dataset = PatchDataset(train_list, args.data_dir, molan, sample,transform_train)
        val_dataset = PatchDataset(val_list, args.data_dir, molan, sample,transform_train)

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
        print(f"[FOLD] {sample} | train={train_list} | val={val_list}")
        print("train size:", len(train_dataset), "val size:", len(val_dataset))
        print("molan_n:", args.molan_n, "img_size:", args.img_size)

        # Model：输出应为 (B, grid_h, grid_w, molan_n)
        # model = CustomModel(
        #     ckpt_path=args.ckpt_path,
        #     model_config=args.model_config,
        #     vocab_size=args.vocab_size,
        #     grid_h=args.grid_h,
        #     grid_w=args.grid_w,
        #     token_dim=args.token_dim,
        #     num_heads=args.num_heads,
        #     num_layers=args.num_layers,
        #     dropout=args.dropout,
        #     molan_n=args.molan_n,
        # ).to(device)
        model = CustomModel(visual_output_dim=1024, num_outputs=num_outputs).to(device)
   

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

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{sample}"
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
                "sample": sample,
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
            model.module.training_status = True
            running_loss = 0.0
            all_preds = []
            all_labels = []
            encodings= []
            global_step = 0
            #train_loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=(global_rank != 0))
            train_loop = tqdm.tqdm(train_loader, desc=f"Train {sample} {epoch+1}/{args.epochs}", leave=False)
            for i, data in train_loop:
                inputs, labels = data[0].to(device, dtype=torch.float16), data[1].to(device, dtype=torch.float16)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    optimizer.zero_grad()
                    outputs,feature = model(inputs,labels,epoch)
                    loss = torch.mean(criterion_ad.lossfun(outputs.to(device, dtype=torch.float32) - labels.to(device, dtype=torch.float32)))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                global_step += 1

                encodings.extend(feature.data.squeeze().cpu().numpy())

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())

            if epoch >= model.module.FDS.start_update:
                encodings, all_labels = torch.from_numpy(np.vstack(encodings)).to(device), torch.from_numpy(np.vstack(all_labels)).to(device)
                model.module.FDS.update_last_epoch_stats(epoch)
                model.module.FDS.update_running_stats(encodings, all_labels.cpu().numpy(), epoch)
            avg_loss = torch.tensor(running_loss / len(train_loader), device=device)

            t_pearson_r_per_biomarker = []
            for i in range(all_labels.shape[1]):  # For each biomarker
                t_r, _ = pearsonr(all_labels[:, i], all_preds[:, i])
                t_pearson_r_per_biomarker.append(t_r)
            t_avg_pearson_r = np.nanmean(t_pearson_r_per_biomarker)

            overall_mse = np.nanmean((all_labels - all_preds) ** 2)
            writer.add_scalar('MSE_train/avg', overall_mse, epoch + 1)
            writer.add_scalar('Pearson_R_train/avg', t_avg_pearson_r, epoch + 1)


            model.eval()
            model.module.training_status = False
            val_loss = 0.0
            all_labels = []
            all_preds = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                #val_loop = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=(global_rank != 0))
                val_loop = tqdm.tqdm(enumerate(val_loader), desc=f"val {sample} {epoch+1}/{args.epochs}", leave=False)
           
                for i, data in val_loop:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs,_ = model(inputs,labels,epoch)
                    all_labels.append(labels)
                    all_preds.append(outputs)

            # Concatenate all tensors
            all_labels = torch.cat(all_labels, dim=0)
            all_preds = torch.cat(all_preds, dim=0)

            writer.add_scalar('MSE_train/avg', overall_mse, epoch + 1)
            writer.add_scalar('Pearson_R_train/avg', t_avg_pearson_r, epoch + 1)

                   
            # -------- save ckpt --------
            is_best = val_mse < best_val_mse
            if is_best:
                best_val_mse = val_mse
                best_epoch = epoch + 1
            save_ckpt(epoch_idx=epoch + 1, is_best=is_best)

        writer.close()








if __name__ == "__main__":
    main()
