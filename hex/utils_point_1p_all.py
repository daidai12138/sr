

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
        df,
        data_dir,
        molan,
        transform=None,
        fixed_ij=(5, 5),      # 固定取点位置
        eps=1e-6,             # 标准化防止除0
        return_stats=False,    # 是否返回 mean/std
    ):
        self.images = df['images'].values
        #self.train_list = list(train_list)
        self.transform = transform

        self.pro_path = join(data_dir, 'PRO')
        self.h5ad_path = join(self.pro_path,  f"adata_all.h5ad")
        self.sample_ids = df['sample_id'].values
        self.img_indexs = df['img_index'].values
        self.pro_indexs = df['pro_index'].values
        
        # molan: pandas index -> numpy indices
        self.molan = molan.index.to_numpy().astype(int)  # (molan_n,), 这里 molan_n=3

        self.fixed_ij = fixed_ij
        self.eps = eps
        self.return_stats = return_stats

        self._adatas = None  # 每个 worker 自己打开

    def _ensure_open(self):
        if self._adatas is None:
            self._adatas = ad.read_h5ad(self.h5ad_path, backed='r')

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_adatas'] = None
        return d

    def __len__(self):
        return len(self.images)

    def _load_grid(self, idx):
        """读取 target 并输出 (10,10,3) 的 grid（未标准化）"""
        self._ensure_open()

        #file_id = self.train_list.index(self.sample_ids[idx])
        row_id = int(self.pro_indexs[idx])

        row = self._adatas.X[row_id]
        y = _to_1d_numpy(row)  # (10*10*7746,)

        pro = y.reshape(10, 10, 7746)
        sub = pro[:, :, self.molan]  # (10,10,3)

        grid = torch.from_numpy(sub).float()  # (10,10,3)
        return grid

    def _per_sample_per_channel_norm(self, grid):
        """
        grid: (10,10,3)
        返回：grid_norm(10,10,3), mean(3,), std(3,)
        """
        # mean/std over H,W for each channel
        mean = grid.mean(dim=(0, 1))  # (3,)
        # unbiased=False：更稳定（尤其点数少时），你这里固定 100 点
        std = grid.std(dim=(0, 1), unbiased=False)  # (3,)
        std = torch.clamp(std, min=self.eps)

        grid_norm = (grid - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        return grid_norm, mean, std

    def __getitem__(self, idx):
        self._ensure_open()

        # image
        image_path = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        # grid (raw)
        grid = self._load_grid(idx)  # (10,10,3)

        # per-sample per-channel normalize
        #grid_norm, mean, std = self._per_sample_per_channel_norm(grid)
        grid_norm = grid


        # point from normalized grid (fixed)


        ii, jj = self.fixed_ij
        point_norm = grid_norm[ii, jj].unsqueeze(0)  # (1,3)

        if self.return_stats:
            return image, grid_norm, point_norm, mean, std
        else:
            return image, grid_norm, point_norm


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

