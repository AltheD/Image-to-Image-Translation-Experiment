"""
评估指标实现：PSNR、SSIM、MAE
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组，处理归一化"""
    if isinstance(tensor, torch.Tensor):
        # 如果是 [-1, 1] 范围，转换到 [0, 1]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        # 转换到 [0, 255] 范围
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        # 转换为numpy
        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
        elif tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        
        tensor = tensor.cpu().numpy()
        tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    
    return tensor


def calculate_psnr(pred, target):
    """
    计算PSNR（峰值信噪比）
    
    Args:
        pred: 预测图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
        target: 真实图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
    
    Returns:
        PSNR值（dB）
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    
    if pred_np.ndim == 4:  # batch
        psnr_values = []
        for i in range(pred_np.shape[0]):
            psnr = peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=255)
            psnr_values.append(psnr)
        return np.mean(psnr_values)
    else:  # single image
        return peak_signal_noise_ratio(target_np, pred_np, data_range=255)


def calculate_ssim(pred, target):
    """
    计算SSIM（结构相似性指数）
    
    Args:
        pred: 预测图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
        target: 真实图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
    
    Returns:
        SSIM值（0-1之间，越接近1越好）
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    
    if pred_np.ndim == 4:  # batch
        ssim_values = []
        for i in range(pred_np.shape[0]):
            # SSIM需要单通道或RGB，如果是RGB需要指定channel_axis
            if pred_np.shape[-1] == 3:
                ssim = structural_similarity(
                    target_np[i], pred_np[i], 
                    data_range=255, 
                    channel_axis=2,
                    win_size=min(7, min(pred_np.shape[1], pred_np.shape[2]))
                )
            else:
                ssim = structural_similarity(
                    target_np[i], pred_np[i], 
                    data_range=255,
                    win_size=min(7, min(pred_np.shape[1], pred_np.shape[2]))
                )
            ssim_values.append(ssim)
        return np.mean(ssim_values)
    else:  # single image
        if pred_np.shape[-1] == 3:
            return structural_similarity(
                target_np, pred_np, 
                data_range=255, 
                channel_axis=2,
                win_size=min(7, min(pred_np.shape[0], pred_np.shape[1]))
            )
        else:
            return structural_similarity(
                target_np, pred_np, 
                data_range=255,
                win_size=min(7, min(pred_np.shape[0], pred_np.shape[1]))
            )


def calculate_mae(pred, target):
    """
    计算MAE（平均绝对误差）
    
    Args:
        pred: 预测图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
        target: 真实图像，tensor [B, C, H, W] 或 [C, H, W]，范围 [-1, 1] 或 [0, 1]
    
    Returns:
        MAE值
    """
    # 确保范围一致
    if pred.min() < 0:  # [-1, 1] -> [0, 1]
        pred = (pred + 1) / 2
    if target.min() < 0:
        target = (target + 1) / 2
    
    mae = torch.mean(torch.abs(pred - target)).item()
    return mae


def evaluate_batch(pred, target):
    """
    批量计算所有评估指标
    
    Args:
        pred: 预测图像，tensor [B, C, H, W]
        target: 真实图像，tensor [B, C, H, W]
    
    Returns:
        dict: {'psnr': float, 'ssim': float, 'mae': float}
    """
    psnr = calculate_psnr(pred, target)
    ssim = calculate_ssim(pred, target)
    mae = calculate_mae(pred, target)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mae': mae
    }









