"""
GKP态Wigner函数生成模块

提供GKP（Gottesman-Kitaev-Preskill）量子态的Wigner函数计算和网格生成功能。
"""

import numpy as np


def gkp_wigner(x, p, delta=0.3, n_peaks=5):
    """
    GKP态的Wigner函数
    
    W_GKP(x, p) ∝ Σ_{s,t∈Z} exp(-(x - s√π)² / (2Δ²)) * exp(-(p - t√π)² / (2Δ²))
    
    参数:
        x: 位置坐标网格
        p: 动量坐标网格
        delta: GKP态压缩参数 (默认0.3)
        n_peaks: 求和的峰数量 (默认5)
    
    返回:
        W: Wigner函数值
    """
    sqrt_pi = np.sqrt(np.pi)
    W = np.zeros_like(x)
    
    for s in range(-n_peaks, n_peaks + 1):
        for t in range(-n_peaks, n_peaks + 1):
            x0 = s * sqrt_pi
            p0 = t * sqrt_pi
            W += np.exp(-((x - x0)**2 + (p - p0)**2) / (2 * delta**2))
    
    # 归一化
    W = W / (2 * np.pi * delta**2)
    return W


def create_gkp_grid(grid_size=64, x_range=(-4, 4), delta=0.3):
    """
    创建GKP态的完整Wigner函数网格
    
    参数:
        grid_size: 网格尺寸 (默认64)
        x_range: 相空间范围 (默认(-4, 4))
        delta: GKP态参数
    
    返回:
        X, P: 坐标网格
        W: Wigner函数值 (float32)
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    p = np.linspace(x_range[0], x_range[1], grid_size)
    X, P = np.meshgrid(x, p)
    W = gkp_wigner(X, P, delta=delta)
    return X, P, W.astype(np.float32)
