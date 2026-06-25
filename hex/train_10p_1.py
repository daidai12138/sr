# tensorboard --logdir ./experiment/runs --port 6006

import os
from os.path import join
import argparse
import math
import random
from datetime import datetime

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
from hex.hex_architecture_10p import CustomModel
from hex.utils_10p import PatchDataset, seed_torch, inverse_per_sample  # noqa: F401


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


def finetuned_state_dict(model: torch.nn.Module):
    """Only save trainable submodules' params/buffers to keep checkpoints smaller."""
    prefixes = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            prefixes.add(n.rsplit(".", 1)[0])

    def keep(k: str) -> bool:
        return any(k == pref or k.startswith(pref + ".") for pref in prefixes)

    return {k: v.detach().cpu() for k, v in model.state_dict().items() if keep(k)}


# ---------------------------
# Trainable stages
# ---------------------------
def set_trainable_for_stage(model: torch.nn.Module, stage: int, unfreeze_last_k: int = 2):
    """
    stage 0: train heads only (regression_head/regression_head1), keep backbone frozen
    stage 1: also unfreeze last K encoder layers (if present) + layer_norm
    """
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # always train heads (CustomModel uses these)
    if hasattr(model, "regression_head"):
        for p in model.regression_head.parameters():
            p.requires_grad = True
    if hasattr(model, "regression_head1"):
        for p in model.regression_head1.parameters():
            p.requires_grad = True

    if stage == 0:
        # optionally unfreeze encoder layer_norm if exists (often stabilizes finetuning)
        try:
            enc = model.visual.beit3.encoder  # type: ignore[attr-defined]
            if hasattr(enc, "layer_norm"):
                for p in enc.layer_norm.parameters():
                    p.requires_grad = True
        except Exception:
            pass
        return

    # stage 1: unfreeze last K layers (best-effort, depending on backbone structure)
    try:
        enc = model.visual.beit3.encoder  # type: ignore[attr-defined]
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
    """Backbone vs heads; avoid weight decay for LN/bias."""

    def is_no_wd(name: str, param: torch.nn.Parameter):
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
        in_head = ("regression_head" in n) or ("regression_head1" in n)
        if in_head:
            (head_no_decay if is_no_wd(n, p) else head_decay).append(p)
        else:
            (backbone_no_decay if is_no_wd(n, p) else backbone_decay).append(p)

    param_groups = []
    if backbone_decay:
        param_groups.append(
            {"params": backbone_decay, "lr": base_lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"}
        )
    if backbone_no_decay:
        param_groups.append(
            {"params": backbone_no_decay, "lr": base_lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"}
        )
    if head_decay:
        param_groups.append({"params": head_decay, "lr": base_lr_head, "weight_decay": weight_decay, "name": "head_decay"})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": base_lr_head, "weight_decay": 0.0, "name": "head_no_decay"})
    return param_groups


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Pearson-correlation loss: 1 - mean(corr) over outputs."""
    pred = pred.float()
    target = target.float()
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    cov = (pred * target).mean(dim=0)
    pred_std = (pred * pred).mean(dim=0).sqrt()
    targ_std = (target * target).mean(dim=0).sqrt()
    corr = cov / (pred_std * targ_std + eps)
    return 1.0 - corr.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # molan config
    parser.add_argument("--molan-n",type=int,default=1)

    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")

    # samples (we loop each sample independently)
    parser.add_argument(
        "--all-list",
        nargs="+",
        default=["N1", "N2", "N3", "N4", "N5", "P1", "P2", "P3", "P4", "P5"],    )
    parser.add_argument(
        "--holdout",
        type=str,
        default="",
        help="if set (e.g. N3), only run this sample id",
    )

    # image / dataloader
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)

    # schedule / finetune
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-last-k", type=int, default=4)

    # optimizer 
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-robust", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # losses
    parser.add_argument("--mse-weight", type=float, default=0.1, help="robust loss + mse_weight * MSE")
    parser.add_argument("--pearson-weight", type=float, default=0.0, help="add pearson correlation loss weight")

    # checkpointing
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint .pth (optional)")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # read molan
    # molan_path = join(args.csv_dir, f"molan_{args.molan_n}_1.csv")
    # molan = pd.read_csv(molan_path, index_col=0)
    # fl_index=np.load('./indices_gt_0p4.npy')
    # molan = molan.iloc[fl_index,:]

    

    #num_outputs = molan.shape[0]    
    num_outputs = 1

    if num_outputs != args.molan_n:
        print(f"[WARN] molan rows={num_outputs} != molan_n={args.molan_n} -> use molan rows.")
        args.molan_n = num_outputs

    # transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )

    # mixed precision helpers
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # pick samples
    samples = args.all_list
    # if args.holdout:
    #     samples = [s for s in samples if s == args.holdout]
    #     if not samples:
    #         raise ValueError(f"--holdout={args.holdout} not in --all-list")

    for sample in samples:
        sample ='P3'
        pro_index = 898
        print("\n" + "=" * 80)
        print(f"[SAMPLE] {sample}")

        # per-sample split of 10x10 patches: choose 10 train points, rest val
        all_points = [(i, j) for i in range(10) for j in range(10)]
        expression_path = join(args.data_dir,'lfq_fi_nofill.csv')
        expression = pd.read_csv(expression_path,sep='\t',index_col=0)
        #print(expression.columns)
        sample_pro = expression.loc[:,expression.columns.str.startswith(sample)].iloc[[pro_index],:].copy()
        #print(sample_pro)
        data_points = []
        target_list = []
        sam_list=[]
        for r,c in all_points:
            sam_i = f'{sample}_{r*10+c+1}'
            if sam_i in sample_pro.columns:
                val = sample_pro.at[sample_pro.index[0], sam_i] 
                # 标量
                if  pd.notna(val):
                    data_points.append((r, c))
                    target_list.append(val)
                    sam_list.append(sam_i)


        
        r_n = len(data_points)
        random.seed(args.seed)
        train_idxs = random.sample(range(r_n), 80)
        val_idxs = [i for i in range(r_n) if i not in train_idxs]
        train_points = [data_points[i] for i in train_idxs]
        train_labels = [target_list[i] for i in train_idxs]

        sam_list_sel = [sam_list[i] for i in train_idxs]
        val_points = [data_points[i] for i in val_idxs]
        val_labels = [target_list[i] for i in val_idxs]
        val_sams   = [sam_list[i] for i in val_idxs]

        train_dataset = PatchDataset(train_points, args.data_dir, pro_index, sample,train_labels, transform_train)
        val_dataset = PatchDataset(val_points, args.data_dir, pro_index, sample,val_labels, transform_val)



        # train_dataset = PatchDataset(train_points, args.data_dir, molan, sample, transform_train)
        # val_dataset = PatchDataset(val_points, args.data_dir, molan, sample, transform_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

        print(f"train points: {train_points}")
        print(f"val points:   {val_points}")
        print("train size:", len(train_dataset), "val size:", len(val_dataset))
        print("molan_n:", args.molan_n, "img_size:", args.img_size)

        # model
        model = CustomModel(visual_output_dim=1024, num_outputs=num_outputs).to(device)

        # robust loss
        criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=args.molan_n,
            float_dtype=torch.float32,
            device=device,
        )

        # stage 0: heads only
        set_trainable_for_stage(model, stage=0, unfreeze_last_k=args.unfreeze_last_k)

        # optimizer & scheduler
        def make_optim_and_sched():
            param_groups = build_param_groups(
                model,
                base_lr_backbone=args.lr_backbone,
                base_lr_head=args.lr_head,
                weight_decay=args.weight_decay,
            )
            opt = torch.optim.AdamW(param_groups)
            opt.add_param_group(
                {"params": list(criterion_ad.parameters()), "lr": args.lr_robust, "weight_decay": 0.0, "name": "robust_params"}
            )
            total_steps = args.epochs * max(1, len(train_loader))
            warmup_steps = args.warmup_epochs * max(1, len(train_loader))
            sch = build_warmup_cosine_scheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps)
            return opt, sch

        optimizer, scheduler = make_optim_and_sched()

        # logging / ckpt dirs
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{sample}"
        writer_dir = join(args.save_dir, "runs", run_name)
        ckpt_dir = join(args.save_dir, "checkpoints", run_name)

        os.makedirs(writer_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        writer = SummaryWriter(writer_dir)

        np.save(join(join(ckpt_dir,'train_point')),train_points)

        # optional graph
        try:
            dummy_img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
            dummy_y = torch.zeros((1, num_outputs), device=device)
            writer.add_graph(model, (dummy_img, dummy_y))
        except Exception as e:
            print(f"[WARN] add_graph failed: {e}")

        # resume
        start_epoch = 0
        best_val_mse = float("inf")
        best_epoch = -1
        global_step = 0

        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] is not None and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_mse = float(ckpt.get("best_val_mse", best_val_mse))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            global_step = int(ckpt.get("global_step", 0))
            print(f"[RESUME] {args.resume} | start_epoch={start_epoch} best_val_mse={best_val_mse:.6f}")

        def save_ckpt(epoch_idx: int, is_best: bool):
            ckpt = {
                "sample": sample,
                "train_points": list(train_points),
                "val_points": list(val_points),
                "epoch": epoch_idx,
                "best_epoch": best_epoch,
                "global_step": global_step,
                "best_val_mse": best_val_mse,
                "model": finetuned_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": (scaler.state_dict() if use_amp else None),
                "args": vars(args),
            }
            torch.save(ckpt, join(ckpt_dir, "checkpoint_last_ft.pth"))
            if is_best:
                torch.save(ckpt, join(ckpt_dir, "checkpoint_best_ft.pth"))

        # ---------------------------
        # training loop
        # ---------------------------
        for epoch in range(start_epoch, args.epochs):
            # stage switch
            if epoch == args.warmup_epochs:
                set_trainable_for_stage(model, stage=1, unfreeze_last_k=args.unfreeze_last_k)
                optimizer, scheduler = make_optim_and_sched()
                print(f"[STAGE] epoch={epoch}: unfreeze last {args.unfreeze_last_k} encoder layers (if available)")

            # ---- train ----
            model.train()
            running_loss = 0.0
            train_labels_np, train_preds_np = [], []
            print("len(train_dataset) =", len(train_loader.dataset))

            pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train {sample} {epoch+1}/{args.epochs}", leave=False)
            for step, (inputs, labels) in pbar:

                inputs = inputs.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs, _features = model(inputs, labels)
                    outputs_f = outputs.float()
                    labels_f = labels.float()
                    labels_f = labels_f.view(-1, 1)  
                    

                    robust = torch.mean(criterion_ad.lossfun(outputs_f - labels_f))
                    mse = F.mse_loss(outputs_f, labels_f)
                    p_loss = pearson_corr_loss(outputs_f, labels_f) if args.pearson_weight > 0 else torch.tensor(0.0, device=device)
                    #loss = robust #+ args.mse_weight * mse + args.pearson_weight * p_loss
                    loss = robust + p_loss
                    #loss = p_loss
                #print('train_loss:',loss.item())
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += float(loss.detach())
                global_step += 1

                train_labels_np.append(labels_f.detach().cpu().numpy())
                train_preds_np.append(outputs_f.detach().cpu().numpy())

            train_labels_np = np.concatenate(train_labels_np, axis=0) if train_labels_np else np.zeros((0, num_outputs))
            train_preds_np = np.concatenate(train_preds_np, axis=0) if train_preds_np else np.zeros((0, num_outputs))

            train_mse = float(np.nanmean((train_labels_np - train_preds_np) ** 2)) if train_labels_np.size else float("nan")
            train_pearsons = []

            if train_labels_np.size:
                for k in range(train_labels_np.shape[1]):
                    r, _ = pearsonr(train_labels_np[:, k], train_preds_np[:, k])
                    train_pearsons.append(r)
            train_pearson_avg = float(np.nanmean(train_pearsons)) if train_pearsons else float("nan")
            train_loss_avg = running_loss / max(1, len(train_loader))

            writer.add_scalar("loss/train", train_loss_avg, epoch + 1)
            writer.add_scalar("MSE/train", train_mse, epoch + 1)
            writer.add_scalar("PearsonR/train", train_pearson_avg, epoch + 1)


            # ---- val ----
            model.eval()
            val_losses = []
            val_labels_np, val_preds_np = [], []

            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val {sample} {epoch+1}/{args.epochs}", leave=False)
                for _step, (inputs, labels) in pbar:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = torch.as_tensor(labels, device=device)

                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs, _ = model(inputs, labels)
                        outputs_f = outputs.float()
                        labels_f = labels.float()
                        labels_f = labels_f.view(-1, 1)  

                        robust = torch.mean(criterion_ad.lossfun(outputs_f - labels_f))
                        mse = F.mse_loss(outputs_f, labels_f)
                        p_loss = pearson_corr_loss(outputs_f, labels_f) if args.pearson_weight > 0 else torch.tensor(0.0, device=device)
                        loss = robust  # + args.mse_weight * mse + args.pearson_weight * p_loss

                    val_losses.append(float(loss.detach()))
                    val_labels_np.append(labels_f.detach().cpu().numpy())
                    val_preds_np.append(outputs_f.detach().cpu().numpy())

            val_labels_np = np.concatenate(val_labels_np, axis=0) if val_labels_np else np.zeros((0, num_outputs))
            val_preds_np = np.concatenate(val_preds_np, axis=0) if val_preds_np else np.zeros((0, num_outputs))
            val_loss_avg = float(np.mean(val_losses)) if val_losses else float("nan")
            val_mse = float(np.nanmean((val_labels_np - val_preds_np) ** 2)) if val_labels_np.size else float("nan")

            val_pearsons = []
            if val_labels_np.size:
                for k in range(val_labels_np.shape[1]):
                    r, _ = pearsonr(val_labels_np[:, k], val_preds_np[:, k])
                    val_pearsons.append(r)
            val_pearson_avg = float(np.nanmean(val_pearsons)) if val_pearsons else float("nan")

            writer.add_scalar("loss/val", val_loss_avg, epoch + 1)
            writer.add_scalar("MSE/val", val_mse, epoch + 1)
            writer.add_scalar("PearsonR/val", val_pearson_avg, epoch + 1)

            # ---- checkpoint ----
            is_best = val_mse < best_val_mse
            if is_best:
                best_val_mse = val_mse
                best_epoch = epoch + 1
            save_ckpt(epoch_idx=epoch + 1, is_best=is_best)

            print(
                f"[{sample}] epoch {epoch+1}/{args.epochs} | "
                f"train_loss={train_loss_avg:.4f} train_mse={train_mse:.6f} train_r={train_pearson_avg:.4f} | "
                f"val_loss={val_loss_avg:.4f} val_mse={val_mse:.6f} val_r={val_pearson_avg:.4f} | "
                f"best_mse={best_val_mse:.6f} (epoch {best_epoch})"
            )

        writer.close()


if __name__ == "__main__":
    main()
