import torch.nn.functional as F

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random

class MultiClassIngredientDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.items = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.output_size = 224
        self.crop_scale = (0.7, 1.0)
        self.crop_ratio = (1.0, 1.0)  #

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_path = os.path.join(self.root_dir, self.items[idx])
        image_path = os.path.join(item_path, 'image.png')

        # === 读取原图 ===
        image = Image.open(image_path).convert('RGB')

        # === 读取4个密度图 ===
        density_maps = []
        for i in range(1, 5):  # c1 to c4
            npy_path = os.path.join(item_path, f'density_c{i}.npy')
            dens = np.load(npy_path)  # 原始尺寸
            dens = torch.tensor(dens, dtype=torch.float32)  # [H, W]
            density_maps.append(dens)
        density = torch.stack(density_maps, dim=0)  # [4, H, W]

        # === 获取随机裁剪参数 ===
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=self.crop_scale, ratio=self.crop_ratio)

        # === 同步裁剪 + 缩放 ===
        image = F.resized_crop(image, i, j, h, w, size=(self.output_size, self.output_size))
        image = F.to_tensor(image)  # [3, 224, 224]

        density = F.resized_crop(density, i, j, h, w, size=(self.output_size, self.output_size))

        return image, density  # image: [3,224,224]  density: [4,224,224]


# 测试可视化
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MultiClassIngredientDataset(r"")  #<-test your data

    print("Total samples:", len(dataset))

    for i in range(len(dataset)):
        img, dens = dataset[i]
        print(f"Image shape: {img.shape}, Density shape: {dens.shape}")
        print("Sum per channel:", [round(d.item(), 2) for d in dens.sum(dim=(1, 2))])

        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        axs[0].imshow(img.permute(1, 2, 0))  # HWC
        axs[0].set_title("Image")
        for j in range(4):
            axs[j + 1].imshow(dens[j], cmap='hot')
            axs[j + 1].set_title(f"Class {j + 1}")
        plt.tight_layout()
        plt.show()
