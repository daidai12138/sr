# infer_sp_1p_3c.py
# 用法示例：
#   python infer_sp_1p_3c.py \
#     --ft-ckpt ./experiment/checkpoints/checkpoint_best_ft.pth \
#     --img-dir ./hex/sample_data/HE \
#     --data-dir ./hex/sample_data/ \
#     --csv-dir ./hex/sample_data \
#     --infer-list N1 P5 \
#     --out-dir ./experiment/infer_out

import os
from os.path import join
import argparse
import math
import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# 项目内模块（与 train_sp_1p_3c.py 一致）
from hex.hex_architecture_1p_3c import CustomModel
from hex.utils_point_1p import PatchDataset, seed_torch


def pearson_r_torch(x, y, eps=1e-12):
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


def build_csv(img_dir: str, sample_list):
    """复刻 train_sp_1p_3c.py 里的 build_csv：生成 images/sample_id/img_index 三列。"""
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


def _strip_module_prefix(state_dict):
    # 兼容 DataParallel 保存的 "module.xxx"
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_finetuned_weights(model: torch.nn.Module, ckpt_path: str):
    """加载训练脚本保存的 checkpoint_best_ft.pth / checkpoint_last_ft.pth."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 训练脚本保存的是 ckpt["ft_model"]（只含 requires_grad=True 的参数）
    if isinstance(ckpt, dict) and "ft_model" in ckpt:
        state = ckpt["ft_model"]
    elif isinstance(ckpt, dict) and "model_finetuned" in ckpt:
        state = ckpt["model_finetuned"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        # 极端情况：ckpt 直接就是 state_dict
        state = ckpt

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[LOAD] finetuned ckpt: {ckpt_path}")
    if missing:
        print(f"  - missing keys (ok if you saved partial ft_model): {len(missing)}")
    if unexpected:
        print(f"  - unexpected keys: {len(unexpected)}")


@torch.no_grad()
def run_inference(model, loader, device, use_amp: bool):
    model.eval()

    preds = []
    gts = []

    for batch in tqdm.tqdm(loader, desc="Infer", leave=False):
        # 训练里 batch = (inputs, labels, points)
        inputs = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
        points = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None

        # points 形状通常是 (B,1,molan_n)
        # 若你的数据集返回 None，可在这里自行构造：
        # if points is None:
        #     points = torch.zeros((inputs.size(0), 1, model.molan_n), device=device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                out = model(inputs, points)  # (B,10,10,molan_n)
        else:
            out = model(inputs, points)

        preds.append(out.detach().cpu().float())

        if labels is not None:
            gts.append(labels.detach().cpu().float())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0) if len(gts) > 0 else None
    return preds, gts


def main():
    parser = argparse.ArgumentParser()

    # 必要参数
    parser.add_argument("--ft-ckpt", type=str, default="./experiment/checkpoints/checkpoint_last_ft.pth",  help="checkpoint_best_ft.pth or checkpoint_last_ft.pth")
    parser.add_argument("--out-dir", type=str, default="./infer_out")

    # 数据参数（和 train_sp_1p_3c.py 对齐）
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--img-dir", type=str, default="./hex/sample_data/HE")
    parser.add_argument("--csv-dir", type=str, default="./hex/sample_data")
    parser.add_argument("--infer-list", nargs="+", default=["N1", "P5"])

    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # molan / 模型结构参数（必须与训练一致）
    parser.add_argument("--molan-n", type=int, default=3)
    parser.add_argument("--ckpt-path", type=str, default="./MUSK/model.safetensors")
    parser.add_argument("--model-config", type=str, default="musk_large_patch16_384")
    parser.add_argument("--vocab-size", type=int, default=64010)
    parser.add_argument("--grid-h", type=int, default=10)
    parser.add_argument("--grid-w", type=int, default=10)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 固定随机种子（可选）
    try:
        seed_torch(args.seed)
    except Exception:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = (torch.cuda.is_available() and (not args.no_amp))

    # 读 molan（训练里也是 molan_{molan_n}_1.csv）
    molan_path = join(args.csv_dir, f"molan_{args.molan_n}_1.csv")

    molan = pd.read_csv(molan_path, index_col=0)
    if molan.shape[0] != args.molan_n:
        print(f"[WARN] molan rows={molan.shape[0]} != molan_n={args.molan_n}，以 molan rows 为准。")
        args.molan_n = int(molan.shape[0])

    # 构建 inference 的 csv（与训练相同的列结构）
    infer_csvs = build_csv(args.img_dir, args.infer_list)

    # 保存 index，方便你把 preds 对回原图（顺序一致：第 i 行 <-> preds[i]）
    index_csv_path = join(args.out_dir, "pred_index.csv")
    infer_csvs.to_csv(index_csv_path, index=False)
    print(f"[SAVE] index csv -> {index_csv_path}")

    # transform（用 val 的）
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    # Dataset / Loader（依赖你项目内 PatchDataset）
    infer_dataset = PatchDataset(infer_csvs, args.data_dir, molan, args.infer_list, transform_val)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # Model（与训练保持一致）
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

    # 加载 finetuned 权重（训练保存的是 ckpt["ft_model"]）
    model.load_state_dict(torch.load(args.ft_ckpt ,map_location='cpu'), strict=False)
    #load_finetuned_weights(model, args.ft_ckpt)

    # 推理
    preds, gts = run_inference(model, infer_loader, device, use_amp=use_amp)

    # 保存 preds
    preds_path = join(args.out_dir, "preds.npz")
    gts_path = join(args.out_dir, "gts.npz")
    np.savez_compressed(preds_path, preds=preds.numpy())
    np.savez_compressed(gts_path, preds=gts.numpy())
    print(f"[SAVE] preds -> {preds_path}  shape={tuple(preds.shape)}")

    # 如果你的 PatchDataset 返回了 labels（训练就是 batch[1] labels :contentReference[oaicite:6]{index=6}），这里顺便算一下指标
    if gts is not None:
        diff = preds - gts
        mse = float((diff * diff).mean().item())
        mae = float(diff.abs().mean().item())
        rmse = math.sqrt(max(0.0, mse))
        pearson = float(pearson_r_torch(preds, gts).mean().item())
        print(f"[METRIC] mse={mse:.6f} rmse={rmse:.6f} mae={mae:.6f} pearson={pearson:.4f}")

        metrics_path = join(args.out_dir, "metrics.json")
        import json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"mse": mse, "rmse": rmse, "mae": mae, "pearson": pearson}, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
