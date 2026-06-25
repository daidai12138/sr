import os
from os.path import join
import pandas as pd
import numpy as np
from PIL import Image
import tqdm
from scipy.stats import pearsonr

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import robust_loss_pytorch

from hex.hex_architecture_3 import CustomModel

from hex.utils import *



# 你原来就有的函数/类：PatchDataset, seed_torch, print_network 等
# 请确保它们在别处已定义或导入
# class PatchDataset(Dataset): ...
# def seed_torch(seed): ...
# def print_network(model): ...
# /home/gao/miniconda3/envs/HEX/bin/python -m hex.train_sp_3









def main():
    # ===== 单卡设置 =====
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank, global_rank, world_size = 0, 0, 1  # 单卡固定
    #seed_torch(42)

    molan_n = 500
    save_dir = "./experiment/"
    data_dir = "./hex/sample_data/"


    img_dir = "./hex/sample_data/HE"
    pro_dir = join(data_dir,'PRO')
    csv_dir = "./hex/sample_data"
    train_list = ['N2','N3','N4','N5','P1','P2','P3','P4']
    val_list   = ['N1','P5']
    # train_list = ['P1']
    # val_list   = ['P1']

    molan = pd.read_csv(join(csv_dir, f'molan_{molan_n}.csv'),index_col =0)

    # ===================== 构建 train_csvs =====================
    train_csvs = []
    for train_sample in train_list:
        train_path_list = os.listdir(join(img_dir, train_sample))
        train_path_list = sorted(train_path_list, key=lambda s: int(os.path.splitext(s)[0].rsplit('_', 1)[1]))

        train_csv = pd.DataFrame()
        # 存完整路径，否则 PatchDataset 里 Image.open 会找不到
        train_csv['images'] = [join(img_dir, train_sample, s) for s in train_path_list]
        train_csv['sample_id'] = train_sample
        # 每张图一行 index（不要用 [sorted(...)] 那种整列只有一个list）
        train_csv['img_index'] = [int(os.path.splitext(s)[0].rsplit("_", 1)[1]) for s in train_path_list]

        train_csvs.append(train_csv)

    train_csvs = pd.concat(train_csvs).reset_index(drop=True)
    train_csvs = train_csvs.sample(frac=0.5, random_state=42).reset_index(drop=True)

    # ===================== 构建 val_csvs（修正变量名） =====================
    val_csvs = []
    for val_sample in val_list:
        val_path_list = os.listdir(join(img_dir, val_sample))
        val_path_list = sorted(val_path_list, key=lambda s: int(os.path.splitext(s)[0].rsplit('_', 1)[1]))

        val_csv = pd.DataFrame()
        val_csv['images'] = [join(img_dir, val_sample, s) for s in val_path_list]
        val_csv['sample_id'] = val_sample
        val_csv['img_index'] = [int(os.path.splitext(s)[0].rsplit("_", 1)[1]) for s in val_path_list]

        val_csvs.append(val_csv)

    val_csvs = pd.concat(val_csvs).reset_index(drop=True)

    # 你原来这里有个 sss 会直接报错，单卡版先删掉/注释掉

    #label_columns = [f'mean_intensity_channel{i}' for i in range(1, 41)]
    img_size = 384
    # transform_train = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    # ])
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])


    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

    train_dataset = PatchDataset(train_csvs, data_dir,molan,train_list ,transform_train,)
    val_dataset   = PatchDataset(val_csvs,   data_dir,molan,val_list, transform_val)

    num_workers = 8
    train_loader = DataLoader(
        train_dataset, batch_size=48, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=48, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )


    num_outputs = molan.shape[0]
    # num_outputs == 500 here, and we want output (B,10,10,500)
    #model = CustomModel(visual_output_dim=1024, out_h=10, out_w=10, out_c=num_outputs).to(device)
    model = CustomModel(molan_n=num_outputs)

    pretrained = False
    if pretrained:
        checkpoint_path = "./sample_checkpoints.pth"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")

    # ======== 冻结/解冻：把 model.module 全改成 model ========
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.visual.beit3.encoder.layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.visual.beit3.encoder.layer_norm.parameters():
        param.requires_grad = True

    for param in model.token_to_grid.parameters():
        param.requires_grad = True


    print_network(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # robust_loss 的 device 建议传 torch.device 或 "cuda:0"
    criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=molan_n, float_dtype=torch.float32, device=device
    )
    optimizer.add_param_group({'params': criterion_ad.parameters(), 'lr': 1e-5, 'name': 'criterion_ad'})

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    num_epochs = 20

    writer_dir = join(save_dir, "runs")
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)

    checkpoint_dir = join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        model.training_status = True

        running_loss = 0.0
        all_preds, all_labels, encodings = [], [], []

        train_loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, data in train_loop:
            inputs = data[0].to(device, dtype=torch.float16, non_blocking=True)
            labels = data[1].to(device, dtype=torch.float16, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # outputs/labels: (B, 10, 10, 500)
                outputs,feature= model(inputs)

                # robust_loss_pytorch AdaptiveLossFunction commonly expects 2D (N, D)
                # Here we treat each spatial location as one sample: N = B*10*10, D = 500
                diff_2d = (outputs - labels).reshape(-1, outputs.shape[-1])
                loss = torch.mean(criterion_ad.lossfun(diff_2d.to(torch.float32)))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # feature: (B, D)
            encodings.extend(feature.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(outputs.detach().cpu().numpy())

        # FDS 部分：同样去掉 module
        if hasattr(model, "FDS") and epoch >= model.FDS.start_update:
            encodings_t = torch.from_numpy(np.vstack(encodings)).to(device)
            labels_t = torch.from_numpy(np.vstack(all_labels)).to(device)
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(encodings_t, labels_t.cpu().numpy(), epoch)

        avg_loss = running_loss / max(1, len(train_loader))

        writer.add_scalar('Loss/train', avg_loss, epoch + 1)

        # all_labels/all_preds: list of (10,10,500) -> (N,500)
        labels_np = np.asarray(all_labels).reshape(-1, num_outputs)
        preds_np = np.asarray(all_preds).reshape(-1, num_outputs)

        mse_per_output = np.nanmean((labels_np - preds_np) ** 2, axis=0)  # (500,)
        for j in range(num_outputs):
            writer.add_scalar(f'MSE_train/{molan.index[j]}', float(mse_per_output[j]), epoch + 1)
        writer.add_scalar('MSE_train/avg', float(np.nanmean(mse_per_output)), epoch + 1)

        # ===================== validation（去掉 all_gather） =====================
        model.eval()
        model.training_status = False

        all_labels_t, all_preds_t = [], []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            val_loop = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
            for i, data in val_loop:
                inputs = data[0].to(device, dtype=torch.float16, non_blocking=True)
                labels = data[1].to(device, dtype=torch.float16, non_blocking=True)
                outputs, _= model(inputs, labels, epoch)
                all_labels_t.append(labels)
                all_preds_t.append(outputs)

        all_labels_t = torch.cat(all_labels_t, dim=0)  # (B,10,10,500)
        all_preds_t  = torch.cat(all_preds_t,  dim=0)  # (B,10,10,500)

        labels_np = all_labels_t.reshape(-1, num_outputs).cpu().numpy()  # (N,500)
        preds_np  = all_preds_t.reshape(-1, num_outputs).cpu().numpy()   # (N,500)

        mse_per_biomarker = np.nanmean((labels_np - preds_np) ** 2, axis=0)  # (500,)
        overall_mse = float(np.nanmean((labels_np - preds_np) ** 2))

        pearson_r_per_biomarker = []
        for k in range(num_outputs):
            r, _ = pearsonr(labels_np[:, k], preds_np[:, k])
            pearson_r_per_biomarker.append(r)
        avg_pearson_r = float(np.nanmean(pearson_r_per_biomarker))

        writer.add_scalar('MSE_val/avg', overall_mse, epoch + 1)
        writer.add_scalar('Pearson_R_val/avg', avg_pearson_r, epoch + 1)
        for k in range(len(mse_per_biomarker)):
            writer.add_scalar(f'MSE_val/{molan.index[k]}', float(mse_per_biomarker[k]), epoch + 1)
            # NOTE: original code used k+1 which is likely a bug; keep k aligned.
            writer.add_scalar(f'Pearson_R_val/{molan.index[k]}', float(pearson_r_per_biomarker[k]), epoch + 1)

        print(f"Epoch {epoch+1}")
        print(f"Average MSE: {overall_mse:.4f}")
        print(f"Average Pearson R: {avg_pearson_r:.4f}")

        # ===================== save（去掉 barrier + model.module） =====================
        save_frequency = 1
        if (epoch + 1) % save_frequency == 0:
            torch.save(model.state_dict(), join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
            print(f"Model weights saved for epoch {epoch + 1}")

    print("Finished Training")


if __name__ == "__main__":
    main()




