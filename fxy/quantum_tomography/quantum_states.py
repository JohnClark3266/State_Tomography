"""
量子态Wigner函数生成模块

提供三种量子态的Wigner函数计算：
- Fock态 (光子数态)
- 相干态
- 猫态 (相干态叠加)
"""

import numpy as np
from scipy.special import factorial, eval_genlaguerre


def fock_wigner(x, p, n=1):
    """
    Fock态 |n⟩ 的Wigner函数
    
    W_n(x, p) = (-1)^n / π * L_n(2(x² + p²)) * exp(-(x² + p²))
    
    其中 L_n 是Laguerre多项式
    
    参数:
        x, p: 相空间坐标
        n: 光子数 (默认1)
    
    返回:
        W: Wigner函数值
    """
    r2 = x**2 + p**2
    # L_n(2r²) - Laguerre多项式
    L_n = eval_genlaguerre(n, 0, 2 * r2)
    W = ((-1)**n / np.pi) * L_n * np.exp(-r2)
    return W


def coherent_wigner(x, p, alpha_real=2.0, alpha_imag=0.0):
    """
    相干态 |α⟩ 的Wigner函数
    
    W_α(x, p) = (1/π) exp(-((x - √2 Re(α))² + (p - √2 Im(α))²))
    
    参数:
        x, p: 相空间坐标
        alpha_real: α的实部 (默认2.0)
        alpha_imag: α的虚部 (默认0.0)
    
    返回:
        W: Wigner函数值
    """
    sqrt2 = np.sqrt(2)
    x0 = sqrt2 * alpha_real
    p0 = sqrt2 * alpha_imag
    W = (1/np.pi) * np.exp(-((x - x0)**2 + (p - p0)**2))
    return W


def cat_wigner(x, p, alpha=2.0, parity='even'):
    """
    猫态的Wigner函数
    
    偶猫态: |cat_+⟩ ∝ |α⟩ + |-α⟩
    奇猫态: |cat_-⟩ ∝ |α⟩ - |-α⟩
    
    参数:
        x, p: 相空间坐标
        alpha: 相干振幅 (实数，沿x轴)
        parity: 'even' (偶猫) 或 'odd' (奇猫)
    
    返回:
        W: Wigner函数值
    """
    sqrt2 = np.sqrt(2)
    
    # 两个相干态分量
    W_plus = coherent_wigner(x, p, alpha, 0)
    W_minus = coherent_wigner(x, p, -alpha, 0)
    
    # 干涉项
    interference = (2/np.pi) * np.exp(-(x**2 + p**2)) * np.cos(2 * sqrt2 * alpha * p)
    
    # 归一化因子
    N2_even = 2 * (1 + np.exp(-2 * alpha**2))
    N2_odd = 2 * (1 - np.exp(-2 * alpha**2))
    
    if parity == 'even':
        W = (W_plus + W_minus + interference * np.exp(-alpha**2)) / N2_even
    else:
        W = (W_plus + W_minus - interference * np.exp(-alpha**2)) / N2_odd
    
    return W


def create_state(grid_size=64, state_type=1, x_range=(-5, 5), **params):
    """
    根据参数选择量子态并生成Wigner函数网格
    
    参数:
        grid_size: 网格尺寸 (默认64)
        state_type: 1=Fock态, 2=相干态, 3=猫态
        x_range: 相空间范围
        **params: 态参数
            - n: Fock态的光子数 (默认1)
            - alpha: 相干态/猫态的振幅 (默认2.0)
            - parity: 猫态的奇偶性 (默认'even')
    
    返回:
        X, P: 坐标网格
        W: Wigner函数值 (float32)
        state_name: 态名称描述
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    p = np.linspace(x_range[0], x_range[1], grid_size)
    X, P = np.meshgrid(x, p)
    
    if state_type == 1:
        # Fock态
        n = params.get('n', 1)
        W = fock_wigner(X, P, n=n)
        state_name = f"Fock态 |{n}⟩"
    elif state_type == 2:
        # 相干态
        alpha = params.get('alpha', 2.0)
        W = coherent_wigner(X, P, alpha_real=alpha, alpha_imag=0)
        state_name = f"相干态 |α={alpha}⟩"
    elif state_type == 3:
        # 猫态
        alpha = params.get('alpha', 2.0)
        parity = params.get('parity', 'even')
        W = cat_wigner(X, P, alpha=alpha, parity=parity)
        parity_cn = "偶" if parity == 'even' else "奇"
        state_name = f"{parity_cn}猫态 (α={alpha})"
    else:
        raise ValueError(f"未知的state_type: {state_type}. 请使用: 1=Fock, 2=相干态, 3=猫态")
    
    return X, P, W.astype(np.float32), state_name


# 态类型常量
FOCK = 1
COHERENT = 2
CAT = 3
