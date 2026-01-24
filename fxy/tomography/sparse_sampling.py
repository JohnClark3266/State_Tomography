"""
稀疏采样与掩码模块

提供稀疏量子态层析所需的采样和数据生成功能。
"""

import numpy as np

try:
    from .gkp_state import create_gkp_grid
    from .noise_model import ExperimentalNoise
except ImportError:
    from gkp_state import create_gkp_grid
    from noise_model import ExperimentalNoise


def create_sparse_input(full_wigner, sampling_mask, noise_model=None):
    """
    从完整Wigner函数创建稀疏输入（可选添加噪声）
    
    参数:
        full_wigner: 完整的Wigner函数 (已包含态畸变)
        sampling_mask: 采样掩码
        noise_model: 噪声模型 (可选)
    
    返回:
        sparse_input: (2, H, W) 数组，包含稀疏值和掩码通道
    """
    if noise_model is not None:
        noisy_wigner = noise_model.apply_measurement_noise(full_wigner, sampling_mask)
        sparse_values = np.where(sampling_mask, noisy_wigner, 0.0)
    else:
        sparse_values = np.where(sampling_mask, full_wigner, 0.0)
    
    mask_channel = sampling_mask.astype(np.float32)
    sparse_input = np.stack([sparse_values, mask_channel], axis=0)
    return sparse_input.astype(np.float32)


def generate_random_mask(grid_size, sampling_ratio):
    """
    生成随机采样掩码
    
    参数:
        grid_size: 网格尺寸
        sampling_ratio: 采样比例 (0-1)
    
    返回:
        mask: 布尔掩码数组
    """
    n_samples = int(grid_size * grid_size * sampling_ratio)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    indices = np.random.choice(grid_size * grid_size, n_samples, replace=False)
    mask.flat[indices] = True
    return mask


def generate_training_data(n_samples=150, grid_size=64, sampling_ratios=(0.05, 0.25),
                          add_noise=False, noise_params=None):
    """
    生成训练数据
    
    逻辑:
    1. Target = 理想态 + 物理畸变 (Loss, Drift)
    2. Input = Target + 测量噪声 (Shot, Readout)
    
    参数:
        n_samples: 样本数量
        grid_size: 网格尺寸
        sampling_ratios: 采样率范围 (min, max)
        add_noise: 是否添加噪声
        noise_params: 噪声参数字典
    
    返回:
        inputs: (N, 2, H, W) 输入数组
        targets: (N, 1, H, W) 目标数组
    """
    inputs = []
    targets = []
    
    # 变化delta参数增加多样性
    deltas = np.random.uniform(0.25, 0.4, n_samples)
    ratios = np.random.uniform(sampling_ratios[0], sampling_ratios[1], n_samples)
    
    # 创建噪声模型
    if add_noise and noise_params is not None:
        noise_model = ExperimentalNoise(**noise_params)
    else:
        noise_model = None
    
    for i in range(n_samples):
        # 1. 产生理想态
        _, _, ideal_wigner = create_gkp_grid(grid_size=grid_size, delta=deltas[i])
        
        # 2. 产生实验态 (Target)
        if noise_model:
            target_wigner = noise_model.apply_state_distortion(ideal_wigner)
        else:
            target_wigner = ideal_wigner.copy()
            
        # 3. 产生测量数据 (Input)
        mask = generate_random_mask(grid_size, ratios[i])
        sparse_input = create_sparse_input(target_wigner, mask, noise_model)
        
        inputs.append(sparse_input)
        targets.append(target_wigner[np.newaxis, :, :])
    
    inputs = np.array(inputs)   # (N, 2, H, W)
    targets = np.array(targets) # (N, 1, H, W)
    
    return inputs, targets
