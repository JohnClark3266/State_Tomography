"""
保真度计算模块

提供从Wigner函数计算量子态保真度的功能。
"""

import numpy as np


def compute_fidelity(pred_wigner, target_wigner, dx=None, dp=None):
    """
    从Wigner函数计算保真度
    
    对于纯态，保真度可以通过Wigner函数的重叠积分近似:
    F ≈ 2π ∫∫ W_pred(x,p) * W_target(x,p) dx dp
    
    参数:
        pred_wigner: 预测/重建的Wigner函数
        target_wigner: 目标Wigner函数
        dx, dp: 积分步长 (可选)
    
    返回:
        fidelity: 归一化保真度 (0-1)
    """
    if dx is None:
        dx = 8.0 / pred_wigner.shape[0]
    if dp is None:
        dp = 8.0 / pred_wigner.shape[1]
    
    overlap = np.sum(pred_wigner * target_wigner) * dx * dp
    fidelity = 2 * np.pi * overlap
    
    # 归一化处理
    norm_pred = np.sum(pred_wigner**2) * dx * dp
    norm_target = np.sum(target_wigner**2) * dx * dp
    
    if norm_pred > 0 and norm_target > 0:
        fidelity_normalized = overlap / np.sqrt(norm_pred * norm_target)
        return np.clip(fidelity_normalized, 0, 1)
    
    return np.clip(fidelity, 0, 1)
