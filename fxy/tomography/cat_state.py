"""
猫态(Cat State)Wigner函数生成模块

猫态是相干态的叠加态，是量子计算和量子纠错中的重要资源。
偶猫态: |cat_+⟩ ∝ |α⟩ + |-α⟩
奇猫态: |cat_-⟩ ∝ |α⟩ - |-α⟩
"""

import numpy as np


def coherent_wigner(x, p, alpha_real, alpha_imag):
    """
    相干态的Wigner函数
    
    W_α(x, p) = (1/π) exp(-((x - √2 Re(α))² + (p - √2 Im(α))²))
    """
    sqrt2 = np.sqrt(2)
    x0 = sqrt2 * alpha_real
    p0 = sqrt2 * alpha_imag
    return (1/np.pi) * np.exp(-((x - x0)**2 + (p - p0)**2))


def cat_wigner(x, p, alpha=2.0, parity='even'):
    """
    猫态的Wigner函数
    
    偶猫态 (|α⟩ + |-α⟩):
    W_+(x,p) ∝ W_α(x,p) + W_{-α}(x,p) + 2cos(2√2 α p) W_0(x,p) exp(-α²)
    
    奇猫态 (|α⟩ - |-α⟩):
    W_-(x,p) ∝ W_α(x,p) + W_{-α}(x,p) - 2cos(2√2 α p) W_0(x,p) exp(-α²)
    
    参数:
        x, p: 相空间坐标
        alpha: 相干振幅 (实数，沿x轴)
        parity: 'even' (偶猫) 或 'odd' (奇猫)
    
    返回:
        W: Wigner函数值
    """
    sqrt2 = np.sqrt(2)
    
    # 两个相干态分量 |α⟩ 和 |-α⟩
    W_plus = coherent_wigner(x, p, alpha, 0)   # |α⟩
    W_minus = coherent_wigner(x, p, -alpha, 0)  # |-α⟩
    
    # 干涉项
    # 来自 ⟨α|-α⟩ 和 ⟨-α|α⟩ 的贡献
    interference = (2/np.pi) * np.exp(-(x**2 + p**2)) * np.cos(2 * sqrt2 * alpha * p)
    
    # 归一化因子
    N2 = 2 * (1 + np.exp(-2 * alpha**2))  # 偶猫
    if parity == 'odd':
        N2 = 2 * (1 - np.exp(-2 * alpha**2))  # 奇猫
    
    # 组合
    if parity == 'even':
        W = (W_plus + W_minus + interference * np.exp(-alpha**2)) / N2
    else:
        W = (W_plus + W_minus - interference * np.exp(-alpha**2)) / N2
    
    return W


def create_cat_grid(grid_size=64, x_range=(-5, 5), alpha=2.0, parity='even'):
    """
    创建猫态的完整Wigner函数网格
    
    参数:
        grid_size: 网格尺寸 (默认64)
        x_range: 相空间范围 (默认(-5, 5))
        alpha: 相干振幅
        parity: 'even' 或 'odd'
    
    返回:
        X, P: 坐标网格
        W: Wigner函数值 (float32)
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    p = np.linspace(x_range[0], x_range[1], grid_size)
    X, P = np.meshgrid(x, p)
    W = cat_wigner(X, P, alpha=alpha, parity=parity)
    return X, P, W.astype(np.float32)
