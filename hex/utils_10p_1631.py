

import os
import random
import torch
import numpy as np
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import anndata as ad


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _to_1d_numpy(x):
    """把 anndata backed 读出来的行（可能是 numpy / matrix / sparse）统一转成 1D numpy array。"""
    if isinstance(x, np.ndarray):
        return x.ravel()
    if hasattr(x, "toarray"):  # sparse
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


class PatchDataset(Dataset):
    """
    返回 (image, grid_norm, point_norm, mean, std)

    - image: (3,H,W)
    - grid_norm: (10,10,3)  每个 sample 单独标准化
    - point_norm: (1,3)     从 grid_norm 固定位置取（默认中心点 5,5）
    - mean/std: (3,)        该 sample 的每通道 mean/std（用于 inverse）
    """

    def __init__(
        self,
        train_list,
        data_dir,
        pro_index,
        sample,
        labels,
        transform=None,
        #fixed_ij=(5, 5),      # 固定取点位置
        eps=1e-6,             # 标准化防止除0
        return_stats=False,    # 是否返回 mean/std
        
    ):

        self.train_list = list(train_list)
        self.transform = transform
        self.sample = sample
        self.img_path = join(data_dir, f'he_all/{sample}.jpg')
        self.eps = eps
        self.return_stats = return_stats
        self.pro_index = pro_index
        self.labels =labels




    def __len__(self):
        return len(self.train_list)


    def __getitem__(self, idx):

        # image
        r,c=self.train_list[idx]
        image_path = self.img_path
        img = Image.open(image_path)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img_w, img_h = img.size

        step_x = img_w // 10
        step_y = img_h // 10

        patch_size = 20  # 切片大小 (20x20)

        center_x = int((c + 0.5) * step_x)
        center_y = int((r + 0.5) * step_y)

        left = max(0, center_x - patch_size // 2)
        upper = max(0, center_y - patch_size // 2)
        right = min(img_w, center_x + patch_size // 2)
        lower = min(img_h, center_y + patch_size // 2)

        patch = img.crop((left, upper, right, lower))

        
        pro = self.labels[idx]

        if self.transform:
            patch = self.transform(patch)


        return patch,pro

        # if self.return_stats:
        #     return image, grid_norm, point_norm, mean, std
        # else:
        #     return image, grid_norm, point_norm


# --------- 训练/推理时如果要 inverse 回原尺度，用这个 ----------
def inverse_per_sample(pred_norm, mean, std):
    """
    pred_norm: (B,10,10,3)  模型输出（标准化空间）
    mean/std:  (B,3) 或 (3,)
    返回 pred_raw: (B,10,10,3)
    """
    if mean.dim() == 1:
        mean = mean.view(1, 1, 1, -1)
        std  = std.view(1, 1, 1, -1)
    else:
        mean = mean.view(-1, 1, 1, mean.shape[-1])
        std  = std.view(-1, 1, 1, std.shape[-1])
    return pred_norm * std + mean

