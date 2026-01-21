"""
实验噪声模拟模块

提供量子光学实验中各种噪声的模拟功能，包括：
- 态畸变（光子损耗、校准漂移）
- 测量噪声（散粒噪声、读出噪声、暗计数）
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .fidelity import compute_fidelity


class ExperimentalNoise:
    """模拟真实量子光学实验中的各种噪声"""
    
    def __init__(self, 
                 detection_efficiency=0.85,     # 探测效率 (影响态本身/损耗)
                 dark_count_rate=0.01,          # 暗计数率 (测量噪声)
                 readout_noise_std=0.02,        # 读出噪声 (测量噪声)
                 shot_noise_scale=0.05,         # 散粒噪声 (测量噪声)
                 calibration_drift=0.01,        # 校准漂移 (态畸变)
                 background_level=0.005,        # 背景噪声 (测量噪声)
                 noise_scale=1.0):              # 全局噪声缩放因子
        
        self.eta = detection_efficiency
        self.dark_count = dark_count_rate
        self.readout_std = readout_noise_std
        self.shot_scale = shot_noise_scale
        self.calib_drift = calibration_drift
        self.bg_level = background_level
        self.noise_scale = noise_scale
    
    def apply_state_distortion(self, wigner_values):
        """
        应用态本身的物理畸变（损耗、漂移）
        这些是实验态"固有"的属性，应该包含在训练目标(Target)中
        """
        distorted = wigner_values.copy()
        H, W = wigner_values.shape
        
        # 1. 探测效率/光子损耗 -> 高斯模糊 (Diffusion)
        if self.eta < 1.0:
            loss_factor = (1.0 - self.eta) * 2.0 * self.noise_scale
            sigma = max(0.1, loss_factor)
            distorted = gaussian_filter(distorted, sigma=sigma)
            distorted *= self.eta
        
        # 2. 系统校准漂移 (位置相关的系统误差)
        if self.calib_drift > 0:
            y, x = np.mgrid[0:H, 0:W]
            ny, nx = y/H, x/W
            drift_field = self.calib_drift * np.sin(nx * np.pi) * np.cos(ny * np.pi)
            distorted += drift_field * self.noise_scale
            
        return distorted

    def apply_measurement_noise(self, wigner_values, mask):
        """
        应用测量过程中的随机噪声（散粒噪声、电子噪声）
        这些是测量的"伪影"，应该只出现在输入(Input)中
        """
        noisy = wigner_values.copy()
        sampled_indices = np.where(mask)
        
        for i, j in zip(*sampled_indices):
            measured = wigner_values[i, j]
            
            # 1. 散粒噪声 (Signal Dependent)
            if measured > 0:
                signal_strength = abs(measured) / (np.max(abs(wigner_values)) + 1e-10)
                shot_noise = np.random.normal(0, self.shot_scale * np.sqrt(signal_strength))
                measured += shot_noise * self.noise_scale
            
            # 2. 读出噪声 (Gaussian)
            readout_noise = np.random.normal(0, self.readout_std)
            measured += readout_noise * self.noise_scale
            
            # 3. 暗计数与背景 (Random)
            dark_noise = np.random.exponential(self.dark_count) * np.random.choice([-1, 1])
            background = np.random.normal(self.bg_level, self.bg_level * 0.3)
            measured += (dark_noise + background) * self.noise_scale
            
            noisy[i, j] = measured
            
        return noisy

    def add_noise(self, wigner_values, mask):
        """兼容接口：同时应用畸变和测量噪声"""
        distorted = self.apply_state_distortion(wigner_values)
        noisy = self.apply_measurement_noise(distorted, mask)
        return noisy
    
    def __str__(self):
        return (f"ExperimentalNoise(η={self.eta:.2f}, "
                f"dark={self.dark_count:.3f}, "
                f"readout={self.readout_std:.3f}, "
                f"shot={self.shot_scale:.3f}, "
                f"scale={self.noise_scale:.2f})")


def calibrate_noise_for_fidelity(ideal_wigner, target_fidelity=0.95, 
                                 base_noise_params=None, max_iterations=20):
    """
    调整噪声参数以达到目标保真度
    
    参数:
        ideal_wigner: 理想Wigner函数
        target_fidelity: 目标保真度 (默认0.95)
        base_noise_params: 基础噪声参数字典
        max_iterations: 最大迭代次数
    
    返回:
        ExperimentalNoise: 校准后的噪声模型
    """
    if base_noise_params is None:
        base_noise_params = {
            'detection_efficiency': 0.85,
            'dark_count_rate': 0.01,
            'readout_noise_std': 0.02,
            'shot_noise_scale': 0.05,
            'calibration_drift': 0.01,
            'background_level': 0.005,
        }
    
    # 二分搜索noise_scale
    low_scale = 0.1
    high_scale = 5.0
    best_scale = 1.0
    
    print(f"\n开始噪声校准，目标保真度: {target_fidelity:.3f}")
    
    for iteration in range(max_iterations):
        mid_scale = (low_scale + high_scale) / 2
        
        noise_model = ExperimentalNoise(**base_noise_params, noise_scale=mid_scale)
        distorted_wigner = noise_model.apply_state_distortion(ideal_wigner.copy())
        fidelity = compute_fidelity(distorted_wigner, ideal_wigner)
        
        print(f"  迭代 {iteration+1}: scale={mid_scale:.3f}, F={fidelity:.4f}")
        
        if abs(fidelity - target_fidelity) < 0.005:
            best_scale = mid_scale
            print(f"✓ 找到合适的噪声强度: scale={best_scale:.3f}, F={fidelity:.4f}")
            break
        
        if fidelity > target_fidelity:
            low_scale = mid_scale
        else:
            high_scale = mid_scale
        
        best_scale = mid_scale
    
    final_noise_params = base_noise_params.copy()
    final_noise_params['noise_scale'] = best_scale
    return ExperimentalNoise(**final_noise_params)
