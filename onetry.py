import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# 从 loss.py 复制 _generate_bl_dl_from_mask 函数
def _generate_bl_dl_from_mask(mask):
    """
    从二值 mask 实时生成 BL (Body Label) 和 DL (Detail Label)
    参考 OpenCV distanceTransform 的 PyTorch 实现
    
    Args:
        mask: (B, 1, H, W), float32, values in [0, 1]
    
    Returns:
        bl_mask: (B, 1, H, W), 内部区域高响应
        dl_mask: (B, 1, H, W), 边界区域高响应
    """
    B, C, H, W = mask.shape
    assert C == 1, "Mask must be single-channel"
    
    # 确保 mask 是二值的（0 或 1）
    binary_mask = (mask > 0.5).float()
    
    # 创建距离图
    dist_map = torch.zeros_like(mask)
    
    for b in range(B):
        # 提取单张 mask
        m = binary_mask[b, 0].cpu().numpy().astype(np.uint8)
        
        # 使用 OpenCV 计算精确距离变换
        try:
            import cv2
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            dist = torch.from_numpy(dist).to(mask.device).float()
        except:
            # fallback
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(m)
            dist = torch.from_numpy(dist).to(mask.device).float()
        
        # 归一化到 [0, 1]
        dist_min = dist.min()
        dist_max = dist.max()
        if dist_max > dist_min:
            dist_norm = (dist - dist_min) / (dist_max - dist_min)
        else:
            dist_norm = dist.clone()
        
        dist_map[b, 0] = dist_norm
    
    # 生成 BL 和 DL
    bl_mask = binary_mask * dist_map          # 内部高
    dl_mask = binary_mask * (1.0 - dist_map)  # 边界高
    
    return bl_mask, dl_mask


def visualize_bl_dl(original_mask, bl_mask, dl_mask, save_path=None):
    """
    可视化原始 mask, BL, DL 的对比图
    """
    # 转换为 numpy 以便可视化
    orig_np = original_mask[0, 0].cpu().numpy()  # (H, W)
    bl_np = bl_mask[0, 0].cpu().numpy()          # (H, W)
    dl_np = dl_mask[0, 0].cpu().numpy()          # (H, W)
    
    # 创建热力图（使用你之前提到的柔和颜色）
    def create_heatmap(data):
        # 归一化到 [0, 255]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        data_uint8 = data_norm.astype(np.uint8)
        
        # 定义柔和的 LUT（深紫灰 → 灰绿 → 米黄 → 淡黄）
        colors_rgb = [
            (80, 40, 100),   # 深紫灰
            (120, 100, 120), # 紫灰 → 灰绿
            (150, 140, 100), # 灰绿 → 米黄
            (255, 240, 100), # 米黄 → 淡黄
        ]
        n_colors = len(colors_rgb)
        lut = np.zeros((256, 3), dtype=np.uint8)
        segment_size = 255.0 / (n_colors - 1)

        for i in range(256):
            seg_idx = min(int(i // segment_size), n_colors - 2)
            t = (i - seg_idx * segment_size) / segment_size
            c0 = colors_rgb[seg_idx]
            c1 = colors_rgb[seg_idx + 1]
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
            lut[i] = [b, g, r]  # BGR

        heatmap = lut[data_uint8]  # (H, W, 3)
        return heatmap

    # 生成热力图
    orig_heatmap = create_heatmap(orig_np)
    bl_heatmap = create_heatmap(bl_np)
    dl_heatmap = create_heatmap(dl_np)

    # 二值 mask（白色）
    orig_binary = (orig_np > 0.5).astype(np.uint8) * 255
    orig_binary_rgb = np.stack([orig_binary] * 3, axis=-1)  # (H, W, 3)

    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(orig_binary_rgb)
    axes[0].set_title('Original Binary Mask')
    axes[0].axis('off')
    
    axes[1].imshow(orig_heatmap)
    axes[1].set_title('Original Mask (Heatmap)')
    axes[1].axis('off')
    
    axes[2].imshow(bl_heatmap)
    axes[2].set_title('BL (Body Label)\n(Internal High)')
    axes[2].axis('off')
    
    axes[3].imshow(dl_heatmap)
    axes[3].set_title('DL (Detail Label)\n(Boundary High)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    plt.show()


def test_with_sample_masks():
    """
    生成一些示例 mask 并可视化 BL/DL
    """
    # 创建几个不同形状的示例 mask
    masks = []
    
    # 示例 1: 圆形
    h, w = 256, 256
    center = (h//2, w//2)
    radius = 80
    y, x = np.ogrid[:h, :w]
    mask1 = ((x - center[1])**2 + (y - center[0])**2 <= radius**2).astype(np.float32)[None, None, ...]  # (1, 1, H, W)
    masks.append(("circle", torch.from_numpy(mask1)))
    
    # 示例 2: 椭圆形
    mask2 = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if ((i - center[0])**2 / (radius**2 * 0.5) + (j - center[1])**2 / (radius**2 * 1.5)) <= 1:
                mask2[i, j] = 1.0
    mask2 = mask2[None, None, ...]
    masks.append(("ellipse", torch.from_numpy(mask2)))
    
    # 示例 3: 不规则形状（类似息肉）
    mask3 = np.zeros((h, w), dtype=np.float32)
    # 主体
    mask3[100:180, 80:170] = 1.0
    # 边界细节
    mask3[95:105, 80:170] = 1.0  # 上边界
    mask3[175:185, 80:170] = 1.0  # 下边界
    mask3[100:180, 75:85] = 1.0   # 左边界
    mask3[100:180, 165:175] = 1.0  # 右边界
    mask3 = mask3[None, None, ...]
    masks.append(("polyp_like", torch.from_numpy(mask3)))
    
    # 生成可视化
    os.makedirs("bl_dl_visualization", exist_ok=True)
    
    for name, mask_tensor in masks:
        print(f"Processing {name}...")
        mask_tensor = mask_tensor.float().cuda() if torch.cuda.is_available() else mask_tensor.float()
        
        bl_mask, dl_mask = _generate_bl_dl_from_mask(mask_tensor)
        
        save_path = f"bl_dl_visualization/{name}_bl_dl_comparison.png"
        visualize_bl_dl(mask_tensor, bl_mask, dl_mask, save_path=save_path)


if __name__ == "__main__":
    # 测试示例
    test_with_sample_masks()
    
    # 或者加载你的真实数据进行测试
    # mask_path = "path/to/your/mask.png"
    # mask_img = cv2.imread(mask_path, 0)  # 灰度图
    # mask_tensor = torch.from_numpy(mask_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).cuda()
    # bl_mask, dl_mask = _generate_bl_dl_from_mask(mask_tensor)
    # visualize_bl_dl(mask_tensor, bl_mask, dl_mask, save_path="your_mask_bl_dl.png")