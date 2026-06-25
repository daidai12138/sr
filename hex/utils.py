
import os
import random
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import logging

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

# class PatchDataset(Dataset):
#     def __init__(self, csv,label_columns, transform=None):
#         self.images = csv['images'].values
#         self.labels = csv[label_columns].values
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         label = self.labels[idx, :]
#         image = Image.open(image_path)
#         if self.transform:
#             image = self.transform(image)
#         return image, label, image_path
    
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import anndata as ad


class PatchDataset(Dataset):
    def __init__(self, df,  data_dir, molan, sample_list,transform=None  ):
        self.images = df["images"].values
        #self.label_columns = label_columns
        self.sample_list = sample_list
        self.transform = transform
        self.pro_path = join(data_dir,'PRO')
        self.he_path = join(data_dir,'HE')
        self.h5ad_paths = [join(self.pro_path,f'{x  }', f"{x}_pro.h5ad") for x in sample_list]
        self.sample_ids = df['sample_id'].values
        self.img_indexs = df['img_index'].values
        self.molan = molan.index.to_numpy() 

        #self.obs_row_ids = np.asarray(obs_row_ids)

        self._adatas = None  # 每个 worker 自己打开

    def _ensure_open(self):
        if self._adatas is None:
            self._adatas = [ad.read_h5ad(p, backed="r") for p in self.h5ad_paths]
    # 关键：DataLoader 多进程会 pickle Dataset，把打开的句柄丢掉，避免跨进程共享句柄
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_adatas"] = None
        return d
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        self._ensure_open() 
        image_path = self.images[idx]
        image = Image.open(image_path)

        file_id = self.sample_list.index(self.sample_ids[idx])
        row_id = self.img_indexs[idx]
        #file_id, row_id = self.obs_row_ids[idx]

        # 读取一行 target（不会加载整个 X）
        y = np.asarray(self._adatas[int(file_id)].X[int(row_id)])
        pro = y.reshape(10,10,7746)
        pro = pro[:,:,]
        p_idx = self.molan.astype(int)             # 确保是 int

        sub = pro[:, :, p_idx] 
        if self.transform:
            image = self.transform(image)

        y = torch.from_numpy(sub).float()
        return image, y



def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    print("\nTrainable parameters:")
    for name, param in net.named_parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
            print(f"{name}, Shape: {param.shape}")

    print('\nTotal number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)