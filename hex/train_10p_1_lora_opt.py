

from __future__ import annotations

import os
from os.path import join
import argparse
import math
import random
from datetime import datetime
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import robust_loss_pytorch
from scipy.stats import pearsonr

# project modules
from hex.hex_architecture_10p_lora_opt import CustomModel
from hex.utils_10p import PatchDataset, seed_torch


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


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    cov = (pred * target).mean(dim=0)
    pred_var = (pred * pred).mean(dim=0)
    targ_var = (target * target).mean(dim=0)

    corr = cov / (torch.sqrt(pred_var + eps) * torch.sqrt(targ_var + eps))
    corr = torch.clamp(corr, -1.0, 1.0)
    return 1.0 - corr.mean()


# ---------------------------
# Trainable stages (non-LoRA)
# ---------------------------
def set_trainable_for_stage(model: torch.nn.Module, stage: int, unfreeze_last_k: int = 2):
    for p in model.parameters():
        p.requires_grad = False

    for n in ["regression_head", "regression_head1"]:
        if hasattr(model, n):
            for p in getattr(model, n).parameters():
                p.requires_grad = True

    if stage == 0:
        return

    try:
        enc = model.visual.beit3.encoder
        if hasattr(enc, "layers") and len(enc.layers) > 0:
            for layer in enc.layers[-unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(enc, "layer_norm"):
            for p in enc.layer_norm.parameters():
                p.requires_grad = True
    except Exception:
        pass


def build_param_groups(model, base_lr_backbone: float, base_lr_head: float, weight_decay: float):
    def is_no_wd(name: str, param: torch.nn.Parameter):
        return (param.ndim == 1) or name.endswith(".bias")

    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_head = ("regression_head" in n) or ("regression_head1" in n)
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
# LoRA
# ---------------------------
class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, torch.nn.Linear):
            raise TypeError("LoRALinear expects a nn.Linear as base.")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if self.r > 0:
            self.lora_A = torch.nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, self.r))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        self.lora_dropout = torch.nn.Dropout(p=dropout) if dropout and dropout > 0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            x_d = self.lora_dropout(x)
            lora = (x_d @ self.lora_A.t()) @ self.lora_B.t()
            out = out + self.scaling * lora
        return out


def _get_parent_module(root: torch.nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_to_linear_modules(
    model: torch.nn.Module,
    target_name_regex: str,
    r: int,
    alpha: float,
    dropout: float,
    verbose: bool = True,
) -> List[str]:
    pattern = re.compile(target_name_regex)
    replaced: List[str] = []
    to_replace: List[str] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and pattern.search(name):
            to_replace.append(name)

    for name in to_replace:
        parent, child_name = _get_parent_module(model, name)
        base = getattr(parent, child_name)
        if isinstance(base, torch.nn.Linear):
            setattr(parent, child_name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)

    if verbose:
        print(f"[LoRA] replaced {len(replaced)} Linear modules with regex='{target_name_regex}'")
        if replaced:
            print("[LoRA] examples:", replaced[:10])
    return replaced


def mark_only_lora_and_heads_trainable(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    for n in ["regression_head", "regression_head1"]:
        if hasattr(model, n):
            for p in getattr(model, n).parameters():
                p.requires_grad = True

    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True


def finetuned_state_dict(model: torch.nn.Module):
    prefixes = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            prefixes.add(n.rsplit(".", 1)[0])

    def keep(k: str) -> bool:
        return any(k == pref or k.startswith(pref + ".") for pref in prefixes)

    return {k: v.detach().cpu() for k, v in model.state_dict().items() if keep(k)}


@torch.no_grad()
def count_params(model: torch.nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_pearson_avg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    rs = []
    for k in range(y_true.shape[1]):
        try:
            r, _ = pearsonr(y_true[:, k], y_pred[:, k])
        except Exception:
            r = np.nan
        rs.append(r)
    return float(np.nanmean(rs)) if rs else float("nan")


# ---------------------------
# dataset builder
# ---------------------------
def build_dataset_for_proindex(
    expression: pd.DataFrame,
    sample: str,
    pro_index: int,
    data_dir: str,
    train_ratio: float,
    seed: int,
    transform_train,
    transform_val,
    batch_size: int,
    num_workers: int,
):
    all_points = [(i, j) for i in range(10) for j in range(10)]
    sample_pro = expression.loc[:, expression.columns.str.startswith(sample)].iloc[[pro_index], :].copy()

    data_points: List[Tuple[int, int]] = []
    target_list: List[float] = []

    for r, c in all_points:
        sam_i = f"{sample}_{r*10+c+1}"
        if sam_i in sample_pro.columns:
            val = sample_pro.at[sample_pro.index[0], sam_i]
            if pd.notna(val):
                data_points.append((r, c))
                target_list.append(float(val))

    r_n = len(data_points)
    if r_n < 5:
        raise RuntimeError(f"Too few valid points found for sample={sample}, pro_index={pro_index}: {r_n}")

    rng = random.Random(seed)
    n_train = max(1, int(round(r_n * train_ratio)))
    n_train = min(n_train, r_n - 1)
    train_idxs = rng.sample(range(r_n), n_train)
    val_idxs = [i for i in range(r_n) if i not in train_idxs]

    train_points = [data_points[i] for i in train_idxs]
    train_labels = [target_list[i] for i in train_idxs]
    val_points = [data_points[i] for i in val_idxs]
    val_labels = [target_list[i] for i in val_idxs]

    train_dataset = PatchDataset(train_points, data_dir, pro_index, sample, train_labels, transform_train)
    val_dataset = PatchDataset(val_points, data_dir, pro_index, sample, val_labels, transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_points": train_points,
        "val_points": val_points,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }


def build_model(args, device, num_outputs=1):
    model = CustomModel(visual_output_dim=1024, num_outputs=num_outputs).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[INIT RESUME] loaded from {args.resume}")

    if args.use_lora:
        for p in model.visual.parameters():
            p.requires_grad = False

        replaced = apply_lora_to_linear_modules(
            model.visual,
            target_name_regex=args.lora_target_regex,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=True,
        )

        if len(replaced) == 0:
            print("[LoRA][WARN] No Linear modules matched. Adjust --lora-target-regex.")

    return model


def train_one_proindex(
    args,
    device,
    expression: pd.DataFrame,
    sample: str,
    pro_index: int,
    base_run_name: str,
):
    num_outputs = 1

    # 每次都重新加载模型
    model = build_model(args, device, num_outputs=num_outputs)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

    ds_info = build_dataset_for_proindex(
        expression=expression,
        sample=sample,
        pro_index=pro_index,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        transform_train=transform_train,
        transform_val=transform_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loader = ds_info["train_loader"]
    val_loader = ds_info["val_loader"]
    train_points = ds_info["train_points"]
    val_points = ds_info["val_points"]

    print("\n" + "=" * 100)
    print(f"[SAMPLE] {sample} | pro_index={pro_index} | train={ds_info['train_size']} val={ds_info['val_size']} | img={args.img_size}")
    print(f"train points: {train_points}")
    print(f"val points:   {val_points}")

    if args.use_robust:
        criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=num_outputs,
            float_dtype=torch.float32,
            device=device,
        )
    else:
        criterion_ad = None

    set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)

    if args.use_lora:
        mark_only_lora_and_heads_trainable(model)

    tr, tot = count_params(model)
    print(f"[PARAMS] trainable={tr:,} / total={tot:,} ({100.0 * tr / tot:.4f}%)")

    def make_optim_and_sched():
        param_groups = build_param_groups(
            model,
            base_lr_backbone=args.lr_backbone,
            base_lr_head=args.lr_head,
            weight_decay=args.weight_decay,
        )
        opt = torch.optim.AdamW(param_groups)

        if criterion_ad is not None:
            opt.add_param_group({
                "params": list(criterion_ad.parameters()),
                "lr": args.lr_robust,
                "weight_decay": 0.0,
                "name": "robust_params"
            })

        total_steps = args.epochs * max(1, len(train_loader))
        warmup_steps = args.warmup_epochs * max(1, len(train_loader))
        sch = build_warmup_cosine_scheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps)
        return opt, sch

    optimizer, scheduler = make_optim_and_sched()

    run_name = f"{base_run_name}_{sample}_p{pro_index}"
    writer_dir = join(args.save_dir, "runs", run_name)
    ckpt_dir = join(args.save_dir, "checkpoints", run_name)
    os.makedirs(writer_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)

    best_val_r = -1e9
    best_epoch = -1
    global_step = 0
    patience = 0
    best_ckpt_path = None

    def save_last_ckpt(epoch_idx: int):
        ckpt = {
            "sample": sample,
            "pro_index": pro_index,
            "train_points": list(train_points),
            "val_points": list(val_points),
            "epoch": epoch_idx,
            "best_epoch": best_epoch,
            "global_step": global_step,
            "best_val_r": best_val_r,
            "model": finetuned_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": (scaler.state_dict() if use_amp else None),
            "args": vars(args),
        }
        torch.save(ckpt, join(ckpt_dir, f"checkpoint_last_p{pro_index}.pth"))

    def save_best_ckpt(epoch_idx: int, best_r_value: float):
        nonlocal best_ckpt_path

        if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
            try:
                os.remove(best_ckpt_path)
            except Exception:
                pass

        ckpt = {
            "sample": sample,
            "pro_index": pro_index,
            "train_points": list(train_points),
            "val_points": list(val_points),
            "epoch": epoch_idx,
            "best_epoch": best_epoch,
            "global_step": global_step,
            "best_val_r": best_r_value,
            "model": finetuned_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": (scaler.state_dict() if use_amp else None),
            "args": vars(args),
        }

        safe_r = f"{best_r_value:.4f}".replace("-", "neg")
        best_ckpt_path = join(ckpt_dir, f"checkpoint_best_p{pro_index}_r{safe_r}.pth")
        torch.save(ckpt, best_ckpt_path)

    for epoch in range(args.epochs):
        if (not args.use_lora) and (epoch == args.warmup_epochs):
            set_trainable_for_stage(model, stage=1, unfreeze_last_k=args.unfreeze_last_k)
            optimizer, scheduler = make_optim_and_sched()
            print(f"[STAGE] epoch={epoch}: unfreeze last {args.unfreeze_last_k} encoder layers (if available)")

        # train
        model.train()
        running_loss = 0.0
        train_labels_np, train_preds_np = [], []

        pbar = tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train p{pro_index} {epoch+1}/{args.epochs}",
            leave=False
        )
        for step, (inputs, labels) in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device).float().view(-1, num_outputs)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs, _features = model(inputs, return_features=True)
                outputs_f = outputs.float()

                mse = F.mse_loss(outputs_f, labels)
                p_loss = pearson_corr_loss(outputs_f, labels) if args.pearson_weight > 0 else torch.tensor(0.0, device=device)

                robust = torch.tensor(0.0, device=device)
                if (criterion_ad is not None) and (args.robust_weight > 0):
                    robust = torch.mean(criterion_ad.lossfun(outputs_f - labels))

                loss = args.mse_weight * mse + args.pearson_weight * p_loss + args.robust_weight * robust

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach())
            global_step += 1

            train_labels_np.append(labels.detach().cpu().numpy())
            train_preds_np.append(outputs_f.detach().cpu().numpy())

        train_labels_np = np.concatenate(train_labels_np, axis=0) if train_labels_np else np.zeros((0, num_outputs))
        train_preds_np = np.concatenate(train_preds_np, axis=0) if train_preds_np else np.zeros((0, num_outputs))

        train_mse = float(np.nanmean((train_labels_np - train_preds_np) ** 2)) if train_labels_np.size else float("nan")
        train_r = compute_pearson_avg(train_labels_np, train_preds_np)
        train_loss_avg = running_loss / max(1, len(train_loader))

        writer.add_scalar("loss/train", train_loss_avg, epoch + 1)
        writer.add_scalar("MSE/train", train_mse, epoch + 1)
        writer.add_scalar("PearsonR/train", train_r, epoch + 1)

        # val
        model.eval()
        val_losses = []
        val_labels_np, val_preds_np = [], []

        with torch.no_grad():
            pbar = tqdm.tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Val p{pro_index} {epoch+1}/{args.epochs}",
                leave=False
            )
            for _step, (inputs, labels) in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device).float().view(-1, num_outputs)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs, _ = model(inputs, return_features=True)
                    outputs_f = outputs.float()

                    mse = F.mse_loss(outputs_f, labels)
                    p_loss = pearson_corr_loss(outputs_f, labels) if args.pearson_weight > 0 else torch.tensor(0.0, device=device)

                    robust = torch.tensor(0.0, device=device)
                    if (criterion_ad is not None) and (args.robust_weight > 0):
                        robust = torch.mean(criterion_ad.lossfun(outputs_f - labels))

                    loss = args.mse_weight * mse + args.pearson_weight * p_loss + args.robust_weight * robust

                val_losses.append(float(loss.detach()))
                val_labels_np.append(labels.detach().cpu().numpy())
                val_preds_np.append(outputs_f.detach().cpu().numpy())

        val_labels_np = np.concatenate(val_labels_np, axis=0) if val_labels_np else np.zeros((0, num_outputs))
        val_preds_np = np.concatenate(val_preds_np, axis=0) if val_preds_np else np.zeros((0, num_outputs))

        val_loss_avg = float(np.mean(val_losses)) if val_losses else float("nan")
        val_mse = float(np.nanmean((val_labels_np - val_preds_np) ** 2)) if val_labels_np.size else float("nan")
        val_r = compute_pearson_avg(val_labels_np, val_preds_np)

        writer.add_scalar("loss/val", val_loss_avg, epoch + 1)
        writer.add_scalar("MSE/val", val_mse, epoch + 1)
        writer.add_scalar("PearsonR/val", val_r, epoch + 1)

        is_best = val_r > best_val_r
        if is_best:
            best_val_r = val_r
            best_epoch = epoch + 1
            patience = 0
            save_best_ckpt(epoch_idx=epoch + 1, best_r_value=best_val_r)
        else:
            patience += 1

        save_last_ckpt(epoch_idx=epoch + 1)

        print(
            f"pro_index={pro_index} | epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss_avg:.4f} train_mse={train_mse:.6f} train_r={train_r:.4f} | "
            f"val_loss={val_loss_avg:.4f} val_mse={val_mse:.6f} val_r={val_r:.4f} | "
            f"best_r={best_val_r:.4f} (epoch {best_epoch}) | patience={patience}/{args.early_stop_patience}"
        )

        if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
            print(f"[EARLY STOP] pro_index={pro_index}, no improvement for {patience} epochs.")
            break

    writer.close()

    print(f"[DONE] pro_index={pro_index}, best_r={best_val_r:.4f}, best_epoch={best_epoch}")
    if best_ckpt_path is not None:
        print(f"[BEST CKPT] {best_ckpt_path}")

    return {
        "pro_index": pro_index,
        "best_r": best_val_r,
        "best_epoch": best_epoch,
        "best_ckpt_path": best_ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")

    parser.add_argument("--sample", type=str, default="P3")
    parser.add_argument("--pro-index", type=int, default=None)
    parser.add_argument("--pro-index-list", type=int, nargs="+", default=[34])
    parser.add_argument("--train-ratio", type=float, default=0.3)

    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-last-k", type=int, default=2)

    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-robust", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--use-robust", action="store_true")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--robust-weight", type=float, default=0.0)
    parser.add_argument("--pearson-weight", type=float, default=0.1)

    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-regex",
        type=str,
        default=r"(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.pro_index_list is not None and len(args.pro_index_list) > 0:
        pro_index_list = args.pro_index_list
    elif args.pro_index is not None:
        pro_index_list = [args.pro_index]
    else:
        raise ValueError("Please provide --pro-index or --pro-index-list")

    expression_path = join(args.data_dir, "lfq_fi_nofill.csv")
    expression = pd.read_csv(expression_path, sep="\t", index_col=0)
    pro_df = pd.read_csv(join(args.data_dir, "molan_1631_1.csv"),  index_col=0)
    pro_index_list = pro_df.index.to_list()[300:1000]

    base_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for pro_index in pro_index_list:
        try:
            result = train_one_proindex(
                args=args,
                device=device,
                expression=expression,
                sample=args.sample,
                pro_index=pro_index,
                base_run_name=base_run_name,
            )
            all_results.append(result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] pro_index={pro_index} failed: {e}")

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    for x in all_results:
        print(
            f"pro_index={x['pro_index']} | best_r={x['best_r']:.4f} | "
            f"best_epoch={x['best_epoch']} | ckpt={x['best_ckpt_path']}"
        )


if __name__ == "__main__":
    main()