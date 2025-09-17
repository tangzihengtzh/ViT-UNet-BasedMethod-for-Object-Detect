import os
import torch
import numpy as np
from model import ViT4CNetSkip
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from tqdm import tqdm

cc_threshold = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weight_path):
    model = ViT4CNetSkip().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)  # [1,3,H,W]


def estimate_count(pred_np):
    results = []
    for i in range(4):
        heatmap = pred_np[i]
        if i < 3:
            total_intensity = heatmap.sum()
            gaussian_sigmas = [3.23, 2.8, 1.09, 6.5]
            single_peak = 2 * np.pi * gaussian_sigmas[i] ** 2
            count = total_intensity / single_peak
        else:
            mask = (heatmap > cc_threshold).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(mask)
            count = num_labels - 1
        results.append(count)
    return results  # float list of len=4

# === 主批量验证函数 ===
def batch_validate(val_dir, weight_path, excel_out="val_summary.xlsx"):
    model = load_model(weight_path)
    all_records = []

    item_dirs = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)) and d.startswith("item")])
    for item in tqdm(item_dirs, desc="Batch validating"):
        folder_path = os.path.join(val_dir, item)
        img_path = os.path.join(folder_path, "image.png")
        count_path = os.path.join(folder_path, "counts.txt")

        if not os.path.exists(img_path) or not os.path.exists(count_path):
            print(f"[Skip] Missing file in {item}")
            continue

        img_tensor = load_image(img_path)
        with open(count_path, 'r') as f:
            real_list = [int(line.strip()) for line in f.readlines()]
        assert len(real_list) == 4, f"{item} counts.txt not valid"

        with torch.no_grad():
            pred_tensor = model(img_tensor).cpu().squeeze(0).numpy()

        pred_list = estimate_count(pred_tensor)

        record = {
            "Sample": item,
            "P1": round(pred_list[0], 1), "R1": real_list[0],
            "P2": round(pred_list[1], 1), "R2": real_list[1],
            "P3": round(pred_list[2], 1), "R3": real_list[2],
            "P4": round(pred_list[3], 1), "R4": real_list[3],
            "Total_P": round(sum(pred_list), 1),
            "Total_R": sum(real_list)
        }
        all_records.append(record)

    df = pd.DataFrame(all_records)
    out_path = os.path.join(val_dir, excel_out)
    df.to_excel(out_path, index=False)
    print(f"\n finished: {out_path}")

if __name__ == "__main__":
    batch_validate(
        val_dir=r"",
        weight_path=r""
    )