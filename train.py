import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ViT4CNetSkip
from dataset import MultiClassIngredientDataset

epochs = 200

# 可视化函数
def save_visualization(img, pred, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img_np = img.permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("Input")
    for i in range(4):
        axs[i+1].imshow(pred[i].detach().cpu().numpy(), cmap='jet')
        axs[i+1].set_title(f"Pred C{i+1}")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close()

# 主训练函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiClassIngredientDataset(r"E:\python_prj\data_gen_for_VIT_SUNET\real_data_no_overlap_v1_train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ViT4CNetSkip().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    save_dir = "train_out/ViT4C_skip_exp3_real_v1_train_rand_cut"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "loss_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,loss\n")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for img, dens in loop:
            img = img.to(device)
            dens = dens.to(device)
            pred = model(img)
            loss = criterion(pred, dens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        # 保存一个样本可视化
        model.eval()
        with torch.no_grad():
            sample_img, sample_dens = dataset[0]
            sample_img = sample_img.unsqueeze(0).to(device)
            sample_pred = model(sample_img)[0]
            save_visualization(sample_img[0].cpu(), sample_pred.cpu(), epoch, os.path.join(save_dir, "viz"))

        # === 保存模型 ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth"))

if __name__ == "__main__":
    train()
