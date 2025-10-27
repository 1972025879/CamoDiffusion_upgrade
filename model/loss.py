import torch
import torch.nn.functional as F
import torch.nn as nn


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()


def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (1 - Tversky) ** gamma


def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return bce + (1 - Tversky) ** gamma


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np

def structure_loss(pred, mask):
    """
    智能多分支损失函数：
    - 单输出: 使用标准 structure_loss
    - 三元组: 
        * edge_branch → 用 DL 标签（边界增强）
        * center_branch → 用 BL 标签（内部增强）  
        * final_branch → 用原始 mask
    所有标签均从 mask 实时计算，无需预生成！
    """
    if not isinstance(pred, (tuple, list)):
        return _structure_loss_standard(pred, mask)
    
    if len(pred) == 3:
        pred_final, pred_edge, pred_center = pred
        # print("启动新损失函数咯！")
        # ✅ 实时从 mask 生成 BL 和 DL
        bl_mask, dl_mask = _generate_bl_dl_from_mask(mask)
        
        loss_final = _structure_loss_standard(pred_final, mask)
        loss_edge = _structure_loss_standard(pred_edge, dl_mask)    # Edge 用 DL
        loss_center = _structure_loss_standard(pred_center, bl_mask) # Center 用 BL
        
        return 1.0 * loss_final + 0.2 * loss_edge + 0.2 * loss_center
    
    raise ValueError(f"Unexpected pred type: {type(pred)}")


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
    
    # 使用 PyTorch 实现 distance transform（近似）
    # 方法：对每个前景像素，计算到最近背景像素的 L2 距离
    # 由于 PyTorch 无直接 distanceTransform，我们用形态学膨胀近似
    
    # 创建距离图（简单但有效的方法）
    dist_map = torch.zeros_like(mask)
    
    for b in range(B):
        # 提取单张 mask
        m = binary_mask[b, 0].cpu().numpy().astype(np.uint8)
        
        # 使用 OpenCV 计算精确距离变换（在 GPU 上不可行，但训练可接受）
        # 如果你希望纯 PyTorch 实现，可用迭代膨胀，但 OpenCV 更准
        try:
            import cv2
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            dist = torch.from_numpy(dist).to(mask.device).float()
        except:
            # fallback: 使用简单欧氏距离（较慢）
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


def _structure_loss_standard(pred, mask):
    """标准 structure_loss"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    return _compute_weighted_bce_iou(pred, mask, weit)


def _compute_weighted_bce_iou(pred, mask, weit):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-8)

    return (wbce + wiou).mean()


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def structure_loss_with_ual(pred, mask):
    return structure_loss(pred, mask) + 0.5 * cal_ual(pred, mask)


class Bce_iou_loss(nn.Module):

    def __init__(self):
        super(Bce_iou_loss, self).__init__()

    def forward(self, pred, mask):
        weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (inter + 1) / (union - inter + 1)

        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        return (weighted_bce + weighted_iou).mean()
