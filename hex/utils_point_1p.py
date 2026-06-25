# import os
# import random
# import torch
# import numpy as np
# from os.path import join
# from PIL import Image
# from torch.utils.data import Dataset
# import anndata as ad

# import numpy as np
# import torch

# @torch.no_grad()
# def compute_train_scaler_global_per_channel(dataset, max_items=None, eps=1e-12, seed=7):
#     """
#     dataset: 你的 PatchDataset（注意：此时不要传 grid_mean/grid_std）
#     返回 mean/std: torch.FloatTensor, shape (3,)
#     """
#     rng = np.random.default_rng(seed)
#     C = 3
#     sum_ = np.zeros((C,), dtype=np.float64)
#     sumsq = np.zeros((C,), dtype=np.float64)
#     count = 0

#     n_total = len(dataset)
#     if max_items is None or max_items >= n_total:
#         indices = range(n_total)
#     else:
#         indices = rng.choice(n_total, size=max_items, replace=False)

#     for i in indices:
#         # 直接读未标准化 grid: (10,10,3)
#         grid = dataset._load_grid(int(i))  # torch tensor
#         g = grid.reshape(-1, C).cpu().numpy().astype(np.float64)  # (100,3)
#         sum_ += g.sum(axis=0)
#         sumsq += (g * g).sum(axis=0)
#         count += g.shape[0]  # +100

#     mean = sum_ / count
#     var = sumsq / count - mean * mean
#     var = np.maximum(var, eps)
#     std = np.sqrt(var)

#     return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

# def seed_torch(seed=7):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# # class PatchDataset(Dataset):
# #     """返回 (image, full_grid, point).

# #     - image: (3,H,W)
# #     - full_grid: (10,10,molan_n)
# #     - point: (1,molan_n) —— 从 full_grid 随机采样一个 (i,j) 位置的向量
# #     """

# #     def __init__(self, df, data_dir, molan, train_list, transform=None):
# #         self.images = df['images'].values
# #         self.train_list = train_list
# #         self.transform = transform

# #         self.pro_path = join(data_dir, 'PRO')
# #         self.h5ad_paths = [join(self.pro_path, f'{x}', f"{x}_pro.h5ad") for x in train_list]

# #         self.sample_ids = df['sample_id'].values
# #         self.img_indexs = df['img_index'].values
# #         self.molan = molan.index.to_numpy()

# #         self._adatas = None  # 每个 worker 自己打开

# #     def _ensure_open(self):
# #         if self._adatas is None:
# #             self._adatas = [ad.read_h5ad(p, backed='r') for p in self.h5ad_paths]

# #     def __getstate__(self):
# #         d = self.__dict__.copy()
# #         d['_adatas'] = None
# #         return d

# #     def __len__(self):
# #         return len(self.images)

# #     def __getitem__(self, idx):
# #         self._ensure_open()
# #         image_path = self.images[idx]
# #         image = Image.open(image_path)

# #         file_id = self.train_list.index(self.sample_ids[idx])
# #         row_id = self.img_indexs[idx]

# #         # 读取一行 target（不会加载整个 X）
# #         y = np.asarray(self._adatas[int(file_id)].X[int(row_id)])

# #         # (10,10,7746) -> (10,10,molan_n)
# #         pro = y.reshape(10, 10, 7746)
# #         p_idx = self.molan.astype(int)
# #         sub = pro[:, :, p_idx]  # (10,10,molan_n)

# #         if self.transform:
# #             image = self.transform(image)

# #         grid = torch.from_numpy(sub).float()  # (10,10,molan_n)

# #         # 随机取一个点作为条件输入： (1,molan_n)
# #         ii = np.random.randint(0, grid.shape[0])
# #         jj = np.random.randint(0, grid.shape[1])
# #         ii, jj = 5, 5
# #         point = grid[ii, jj].unsqueeze(0)

# #         return image, grid, point

# class PatchDataset(Dataset):
#     def __init__(self, df, data_dir, molan, train_list, transform=None,
#                  grid_mean=None, grid_std=None, eps=1e-6, fixed_center=True):
#         ...
#         self.grid_mean = grid_mean  # torch tensor (3,)
#         self.grid_std  = grid_std   # torch tensor (3,)
#         self.eps = eps
#         self.fixed_center = fixed_center
#         ...

#     def _scale_grid(self, grid: torch.Tensor):
#         if self.grid_mean is None or self.grid_std is None:
#             return grid
#         mean = self.grid_mean.view(1, 1, -1)
#         std  = torch.clamp(self.grid_std.view(1, 1, -1), min=self.eps)
#         return (grid - mean) / std

#     def __getitem__(self, idx):
#         self._ensure_open()
#         image = Image.open(self.images[idx])
#         if self.transform:
#             image = self.transform(image)

#         grid = self._load_grid(idx)      # (10,10,3) 未标准化
#         grid = self._scale_grid(grid)    # ✅ 标准化到 (10,10,3)

#         if self.fixed_center:
#             ii, jj = 5, 5
#         else:
#             ii = np.random.randint(0, grid.shape[0])
#             jj = np.random.randint(0, grid.shape[1])

#         point = grid[ii, jj].unsqueeze(0)  # ✅ point 也在标准化空间

#         return image, grid, point

# def print_network(net):
#     num_params = 0
#     num_params_train = 0
#     print(net)

#     print('\nTrainable parameters:')
#     for name, param in net.named_parameters():
#         n = param.numel()
#         num_params += n
#         if param.requires_grad:
#             num_params_train += n
#             print(f"{name}, Shape: {param.shape}")

#     print(f"\nTotal number of parameters: {num_params}")
#     print(f"Total number of trainable parameters: {num_params_train}")

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
        train_list,
        transform=None,
        fixed_ij=(5, 5),      # 固定取点位置
        eps=1e-6,             # 标准化防止除0
        return_stats=False,    # 是否返回 mean/std
    ):
        self.images = df['images'].values
        self.train_list = list(train_list)
        self.transform = transform

        self.pro_path = join(data_dir, 'PRO')
        self.h5ad_paths = [join(self.pro_path, f'{x}', f"{x}_pro.h5ad") for x in self.train_list]

        self.sample_ids = df['sample_id'].values
        self.img_indexs = df['img_index'].values

        # molan: pandas index -> numpy indices
        self.molan = molan.index.to_numpy().astype(int)  # (molan_n,), 这里 molan_n=3

        self.fixed_ij = fixed_ij
        self.eps = eps
        self.return_stats = return_stats

        self._adatas = None  # 每个 worker 自己打开

    def _ensure_open(self):
        if self._adatas is None:
            self._adatas = [ad.read_h5ad(p, backed='r') for p in self.h5ad_paths]

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_adatas'] = None
        return d

    def __len__(self):
        return len(self.images)

    def _load_grid(self, idx):
        """读取 target 并输出 (10,10,3) 的 grid（未标准化）"""
        self._ensure_open()

        file_id = self.train_list.index(self.sample_ids[idx])
        row_id = int(self.img_indexs[idx])

        row = self._adatas[int(file_id)].X[row_id]
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

