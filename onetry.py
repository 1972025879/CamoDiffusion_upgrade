from PIL import Image
import numpy as np
import os

# 替换成你新数据集的一个 GT 路径（随便选一个）
gt_path = "/home/4T/wuhao_zjc/ab_new_document/Datasets/EORSSD-dataset/test-labels/0004.png"  # ← 改这里！

# 1. 读取图像
gt_pil = Image.open(gt_path)

# 2. 打印基本信息
print("✅ Mode (格式):", gt_pil.mode)        # 常见: 'L' (灰度), '1' (二值), 'RGB', 'P' (调色板)
print("✅ Size:", gt_pil.size)
print("✅ Format:", gt_pil.format)

# 3. 转为 numpy，看数值
gt_np = np.array(gt_pil)
print("✅ NumPy dtype:", gt_np.dtype)
print("✅ Min pixel:", gt_np.min())
print("✅ Max pixel:", gt_np.max())
print("✅ Unique values:", np.unique(gt_np)[:10])  # 看前10个唯一值

# 4. 判断是否全黑
if gt_np.max() == 0:
    print("⚠️  Warning: This GT is all black (max=0)!")

# 5. 如果是三通道，看是否每个通道一样（伪灰度）
if len(gt_np.shape) == 3:
    if np.all(gt_np[:, :, 0] == gt_np[:, :, 1]) and np.all(gt_np[:, :, 0] == gt_np[:, :, 2]):
        print("ℹ️  It's a pseudo-grayscale RGB image.")
    else:
        print("⚠️  True RGB mask — might cause issues!")