from __future__ import annotations

import os
from os.path import join
import argparse
import math
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# project modules
from hex.hex_architecture_10p_lora_opt import CustomModel
from hex.utils_10p import PatchDataset, seed_torch


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


# ---------------------------
# Metrics
# ---------------------------
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
# Build test set
# ---------------------------
def build_test_points(
    expression: pd.DataFrame,
    sample: str,
    pro_index: int,
    split: str,
    ckpt: Dict,
):
    sample_pro = expression.loc[:, expression.columns.str.startswith(sample)].iloc[[pro_index], :].copy()

    all_points = [(i, j) for i in range(10) for j in range(10)]
    all_data_points = []
    all_labels = []
    all_names = []

    for r, c in all_points:
        sam_i = f"{sample}_{r * 10 + c + 1}"
        if sam_i in sample_pro.columns:
            val = sample_pro.at[sample_pro.index[0], sam_i]
            if pd.notna(val):
                all_data_points.append((r, c))
                all_labels.append(float(val))
                all_names.append(sam_i)

    if split == "all":
        return all_data_points, all_labels, all_names

    ckpt_train_points = ckpt.get("train_points", [])
    ckpt_val_points = ckpt.get("val_points", [])

    point_set = set(tuple(x) for x in (ckpt_val_points if split == "val" else ckpt_train_points))

    data_points = []
    labels = []
    names = []

    for (pt, lb, nm) in zip(all_data_points, all_labels, all_names):
        if tuple(pt) in point_set:
            data_points.append(pt)
            labels.append(lb)
            names.append(nm)

    return data_points, labels, names


# ---------------------------
# Test
# ---------------------------
@torch.no_grad()
def run_test(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device).float().view(-1, 1)

        outputs, _ = model(inputs, return_features=True)
        outputs = outputs.float()

        preds_all.append(outputs.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0, 1))
    labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, 1))

    mse = float(np.nanmean((labels_all - preds_all) ** 2)) if labels_all.size else float("nan")
    pearson_r = compute_pearson_avg(labels_all, preds_all)

    return labels_all, preds_all, mse, pearson_r
from pathlib import Path
import re
def find_best_ckpt(ckpt_root, sample, protein):

    ckpt_root = Path(ckpt_root)

    # 匹配文件夹名：*_P3_p86
    folder_pattern = f"20260311_212520_{sample}_{protein}"
    # 匹配文件名：checkpoint_best_p86_r0.5002.pth
    file_pattern = re.compile(
        rf"^checkpoint_best_{re.escape(protein)}_r([0-9]*\.?[0-9]+)\.pth$"
    )

    candidates = []

    for folder in ckpt_root.glob(folder_pattern):
        if folder.is_dir():
            for file in folder.glob(f"checkpoint_best_{protein}_r*.pth"):
                m = file_pattern.match(file.name)
                if m:
                    corr = float(m.group(1))
                    candidates.append((corr, file))

    if not candidates:
        raise FileNotFoundError(
            f"在 {ckpt_root} 下没有找到 sample={sample}, protein={protein} 对应的 checkpoint_best 文件"
        )

    # 取相关性最高的那个
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_corr, best_file = candidates[0]

    print(f"Auto selected checkpoint: {best_file}")
    print(f"Correlation: {best_corr:.4f}")
    return str(best_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=r"E:\A_ligao_model\HEX\experiment\checkpoints",
        help="root directory of checkpoints"
    )
    #parser.add_argument("--protein", type=str, default="p2202", help="protein name, e.g. p86")
    parser.add_argument("--data-dir", type=str, default="./hex/sample_data/")
    parser.add_argument("--sample", type=str, default='P3')
    parser.add_argument("--pro-index", type=int, default=2268)

    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--save-csv", type=str, default="")

    args = parser.parse_args()
    protein_name = f"p{args.pro_index}"
    args.ckpt = find_best_ckpt(args.ckpt_root, args.sample, protein_name)
    seed_torch(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    
    sample = args.sample if args.sample is not None else ckpt.get("sample", ckpt_args.get("sample"))
    pro_index = args.pro_index if args.pro_index is not None else ckpt.get("pro_index", ckpt_args.get("pro_index"))

    if sample is None or pro_index is None:
        raise ValueError("Cannot determine sample/pro_index. Please provide --sample and --pro-index manually.")

    print(f"[INFO] sample={sample}, pro_index={pro_index}, split={args.split}")

    expression_path = join(args.data_dir, "lfq_fi_nofill.csv")
    expression = pd.read_csv(expression_path, sep="\t", index_col=0)

    model = CustomModel(visual_output_dim=1024, num_outputs=1).to(device)

    use_lora = ckpt_args.get("use_lora", False)
    if use_lora:
        replaced = apply_lora_to_linear_modules(
            model.visual,
            target_name_regex=ckpt_args.get("lora_target_regex", r"(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)"),
            r=ckpt_args.get("lora_r", 8),
            alpha=ckpt_args.get("lora_alpha", 16.0),
            dropout=ckpt_args.get("lora_dropout", 0.05),
            verbose=True,
        )
        if len(replaced) == 0:
            print("[LoRA][WARN] No modules matched when rebuilding LoRA structure.")

    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[LOAD] missing examples:", missing[:10])
    if len(unexpected) > 0:
        print("[LOAD] unexpected examples:", unexpected[:10])

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    test_points, test_labels, test_names = build_test_points(
        expression=expression,
        sample=sample,
        pro_index=pro_index,
        split=args.split,
        ckpt=ckpt,
    )

    if len(test_points) == 0:
        raise RuntimeError(f"No valid test points found for split={args.split}")

    print(f"[INFO] num_test_points={len(test_points)}")
    print(f"[INFO] test_points={test_points}")

    test_dataset = PatchDataset(
        test_points,
        args.data_dir,
        pro_index,
        sample,
        test_labels,
        transform_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    labels_np, preds_np, mse, pearson_r = run_test(model, test_loader, device)

    print("\n" + "=" * 80)
    print(f"[RESULT] sample={sample} | pro_index={pro_index} | split={args.split}")
    print(f"[RESULT] MSE = {mse:.6f}")
    print(f"[RESULT] Pearson r = {pearson_r:.6f}")

    result_df = pd.DataFrame({
        "sample_name": test_names,
        "point": [f"({r},{c})" for r, c in test_points],
        "y_true": labels_np.reshape(-1),
        "y_pred": preds_np.reshape(-1),
        "abs_error": np.abs(labels_np.reshape(-1) - preds_np.reshape(-1)),
        "sq_error": (labels_np.reshape(-1) - preds_np.reshape(-1)) ** 2,
    })

    print("\n[HEAD]")
    print(result_df.head())

    save_csv = args.save_csv
    if save_csv == "":
        ckpt_dir = os.path.dirname(args.ckpt)
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_csv = join(ckpt_dir, f"{ckpt_name}_test_{args.split}.csv")

    result_df.to_csv(save_csv, index=False)
    print(f"\n[SAVED] {save_csv}")


if __name__ == "__main__":
    main()