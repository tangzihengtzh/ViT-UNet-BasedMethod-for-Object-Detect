# ViT-UNet Based Method for Object Counting/Detection

This project provides a PyTorch implementation of a hybrid **Vision Transformer (ViT) encoder + UNet decoder** framework.  
It is designed for **dense small-object scenarios** where conventional detection frameworks often fail, such as:

- Counting agricultural seeds  
- Vehicle counting in aerial imagery  
- Other crowded object environments  

The method predicts **density maps** of objects, enabling accurate counting and visualization without the need for explicit bounding boxes.

---

## Project Structure
```

├─ dataset.py        # Dataset preparation and loading
├─ model.py          # ViT + UNet model definition
├─ train.py          # Training script
├─ val.py            # Validation and inference script
├─ requirements.txt  # Dependencies
└─ outputs/          # Example outputs and results

```

---

## Visualization
<img width="1025" height="443" alt="image" src="https://github.com/user-attachments/assets/16fa10ea-27b0-42a7-8640-d334c9a49316" />
framework of the proposed method

<img width="1816" height="377" alt="image" src="https://github.com/user-attachments/assets/3f83c0e6-c6b8-4fda-9605-4640b2466124" />  
<img width="948" height="594" alt="image" src="https://github.com/user-attachments/assets/c8997cab-ddfa-4879-b8d4-a7b9a1bed06e" />  

*Visualization results of the proposed method*

---

# 基于 ViT-UNet 的目标计数方法

本项目基于 PyTorch 实现了一种 **ViT 编码器 + UNet 解码器** 的混合框架。  
该方法主要面向**密集小目标场景**，在传统检测框架难以适用的情况下表现优越，典型应用包括：

- 农作物种子计数  
- 航拍图像中的车辆计数  
- 其他密集目标的检测与计数  

通过预测**密度图**，实现精确的计数与可视化，无需使用传统的目标边界框。

---

## 项目结构
```

├─ dataset.py        # 数据集准备与加载
├─ model.py          # ViT + UNet 模型定义
├─ train.py          # 训练脚本
├─ val.py            # 验证与推理脚本
├─ requirements.txt  # 依赖库
└─ outputs/          # 示例输出


