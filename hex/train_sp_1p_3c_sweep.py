# overfit_one_batch_sp_1p_3c.py
# 用法示例：
#   python overfit_one_batch_sp_1p_3c.py --train-list N2 --batch-size 1 --steps 3000 --lr 1e-3 --dropout 0.0
# tensorboard --logdir ./experiment/runs --port 6006

import os
from os.path import join
import argparse
import math
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

# 你项目内模块
from hex.hex_architecture_1p_3c import CustomModel
from hex.utils_point_1p import PatchDataset, seed_torch#, print_network


def pearson_r_torch(x, y, eps=1e-6):
    # x,y: (B,H,W,C)
    b, _, _, c = x.shape
    x = x.permute(0, 3, 1, 2).reshape(b, c, -1)
    y = y.permute(0, 3, 1, 2).reshape(b, c, -1)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    cov = (x * y).mean(dim=-1)
    stdx = torch.sqrt((x * x).mean(dim=-1))
    stdy = torch.sqrt((y * y).mean(dim=-1))
    return cov / (stdx * stdy + eps)  # (B,C)


def set_trainable_for_overfit(model: torch.nn.Module, train_ln: bool = True, unfreeze_last_k: int = 0):
    """
    overfit 单 batch：先只训 head（token_to_grid/film/out_head/out_head_2）+ 可选 encoder layer_norm
    如你想更强拟合，可把 unfreeze_last_k>0 解冻 backbone 最后K层
    """
    for p in model.parameters():
        p.requires_grad = False

    # head
    for name in ["token_to_grid", "film", "out_head", "out_head_2"]:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True

    # encoder LN（常见更稳）
    if train_ln and hasattr(model, "visual") and hasattr(model.visual, "beit3"):
        enc = model.visual.beit3.encoder
        if hasattr(enc, "layer_norm"):
            for p in enc.layer_norm.parameters():
                p.requires_grad = True

    # 可选：解冻最后K层（更强拟合，但更慢）
    if unfreeze_last_k > 0 and hasattr(model, "visual") and hasattr(model.visual, "beit3"):
        enc = model.visual.beit3.encoder
        if hasattr(enc, "layers"):
            for layer in enc.layers[-unfreeze_last_k:]:
                for p in layer.parameters():
                    p.requires_grad = True


def build_csv(img_dir: str, sample_list):
    """生成 images/sample_id/img_index 三列"""
    all_csvs = []
    for sample_id in sample_list:
        folder = join(img_dir, sample_id)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"sample folder not found: {folder}")
        path_list = os.listdir(folder)
        path_list = sorted(path_list, key=lambda s: int(os.path.splitext(s)[0].rsplit("_", 1)[1]))
        df = pd.DataFrame()
        df["images"] = [join(folder, s) for s in path_list]
        df["sample_id"] = sample_id
        df["img_index"] = [int(os.path.splitext(s)[0].rsplit("_", 1)[1]) for s in path_list]
        all_csvs.append(df)
    return pd.concat(all_csvs).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./experiment/")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--img-dir", type=str, default="./hex/sample_data/HE")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")
    parser.add_argument("--train-list", nargs="+", default=["N2"])
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    # 模型
    parser.add_argument("--molan-n", type=int, default=3)
    parser.add_argument("--ckpt-path", type=str, default="./MUSK/model.safetensors")
    parser.add_argument("--model-config", type=str, default="musk_large_patch16_384")
    parser.add_argument("--vocab-size", type=int, default=64010)
    parser.add_argument("--grid-h", type=int, default=10)
    parser.add_argument("--grid-w", type=int, default=10)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    # overfit 超参
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-ln", action="store_true")
    parser.add_argument("--unfreeze-last-k", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = (torch.cuda.is_available() and (not args.no_amp))
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)
    try:
        seed_torch(args.seed)
    except Exception:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # molan
    molan_path = join(args.csv_dir, f"molan_{args.molan_n}_1.csv")
    molan = pd.read_csv(molan_path, index_col=0)
    if molan.shape[0] != args.molan_n:
        print(f"[WARN] molan rows={molan.shape[0]} != molan_n={args.molan_n}，以 molan rows 为准。")
        args.molan_n = int(molan.shape[0])

    # dataset / loader：overfit 不要 shuffle
    train_csvs = build_csv(args.img_dir, args.train_list)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    train_dataset = PatchDataset(train_csvs, args.data_dir, molan, args.train_list, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,            # 关键：固定顺序
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,          # 关键：别丢数据
    )

    print("train size:", len(train_dataset))
    print("molan_n:", args.molan_n, "img_size:", args.img_size)

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
        molan_n=args.molan_n,
    ).to(device)

    try:
        print_network(model)
    except Exception:
        pass

    set_trainable_for_overfit(model, train_ln=args.train_ln, unfreeze_last_k=args.unfreeze_last_k)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[OVERFIT] trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # tensorboard
    run_name = "overfit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer_dir = join(args.save_dir, "runs", run_name)
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)

    # 取固定 batch（只取一次）
    fixed_batch = next(iter(train_loader))
    fixed_inputs = fixed_batch[0].to(device, non_blocking=True)
    fixed_labels = fixed_batch[1].to(device, non_blocking=True)
    fixed_points = fixed_batch[2].to(device, non_blocking=True)

    model.train()
    pbar = tqdm.tqdm(range(args.steps), desc="OverfitOneBatch", leave=True)

    for step in pbar:
        # 防止你 model 里 hook 的 tokens 复用旧 batch（如果有这个属性）
        if hasattr(model, "_last_tokens"):
            model._last_tokens = None

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(fixed_inputs, fixed_points)  # (B,10,10,C)
            loss = F.smooth_l1_loss(outputs, fixed_labels, beta=0.1)

        scaler.scale(loss).backward()

        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if (step % args.log_every) == 0 or step == args.steps - 1:
            with torch.no_grad():
                pred = outputs.float()
                gt = fixed_labels.float()
                r = pearson_r_torch(pred, gt).mean().item()
                pred_std = pred.std(dim=(1, 2)).mean(0).cpu().numpy()
                gt_std = gt.std(dim=(1, 2)).mean(0).cpu().numpy()
                pred_mean = pred.mean(dim=(1, 2)).mean(0).cpu().numpy()
                gt_mean = gt.mean(dim=(1, 2)).mean(0).cpu().numpy()

            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", pearson=f"{r:.4f}", lr=f"{lr_now:.2e}")

            writer.add_scalar("overfit/loss", float(loss.item()), step)
            writer.add_scalar("overfit/pearson", float(r), step)

            # 这两行帮助你判断“输出塌缩/均值偏移”有没有在改善
            print(f"[{step:05d}] loss={loss.item():.6f} pearson={r:.4f}")
            print("  pred_std:", pred_std, "gt_std:", gt_std)
            print("  pred_mean:", pred_mean, "gt_mean:", gt_mean)

    writer.close()
    print(f"[DONE] TensorBoard: {writer_dir}")


if __name__ == "__main__":
    main()
