"""
实验噪声模拟模块

模拟量子光学实验中的各种噪声：
- 态畸变：光子损耗（高斯模糊）、校准漂移
- 测量噪声：散粒噪声、读出噪声、暗计数、背景噪声
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class ExperimentalNoise:
    """实验噪声模拟器"""
    
    def __init__(self,
                 detection_efficiency=0.90,     # 探测效率
                 dark_count_rate=0.005,         # 暗计数率
                 readout_noise_std=0.015,       # 读出噪声标准差
                 shot_noise_scale=0.03,         # 散粒噪声比例
                 calibration_drift=0.005,       # 校准漂移
                 background_level=0.003,        # 背景噪声
                 noise_scale=1.0):              # 全局噪声缩放
        
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
        这些是实验态"固有"的属性
        """
        distorted = wigner_values.copy()
        H, W = wigner_values.shape
        
        # 1. 光子损耗 -> 高斯模糊
        if self.eta < 1.0:
            loss_factor = (1.0 - self.eta) * 2.0 * self.noise_scale
            sigma = max(0.1, loss_factor)
            distorted = gaussian_filter(distorted, sigma=sigma)
            distorted *= self.eta
        
        # 2. 校准漂移
        if self.calib_drift > 0:
            y, x = np.mgrid[0:H, 0:W]
            ny, nx = y/H, x/W
            drift_field = self.calib_drift * np.sin(nx * np.pi) * np.cos(ny * np.pi)
            distorted += drift_field * self.noise_scale
        
        return distorted
    
    def apply_measurement_noise(self, wigner_values, mask):
        """
        应用测量过程中的随机噪声
        只作用于采样点
        """
        noisy = wigner_values.copy()
        sampled_indices = np.where(mask)
        
        for i, j in zip(*sampled_indices):
            measured = wigner_values[i, j]
            
            # 1. 散粒噪声
            if measured != 0:
                signal_strength = abs(measured) / (np.max(np.abs(wigner_values)) + 1e-10)
                shot_noise = np.random.normal(0, self.shot_scale * np.sqrt(signal_strength))
                measured += shot_noise * self.noise_scale
            
            # 2. 读出噪声
            readout_noise = np.random.normal(0, self.readout_std)
            measured += readout_noise * self.noise_scale
            
            # 3. 暗计数与背景
            dark_noise = np.random.exponential(self.dark_count) * np.random.choice([-1, 1])
            background = np.random.normal(self.bg_level, self.bg_level * 0.3)
            measured += (dark_noise + background) * self.noise_scale
            
            noisy[i, j] = measured
        
        return noisy


def compute_fidelity(pred_wigner, target_wigner):
    """
    计算Wigner函数保真度
    
    使用归一化重叠积分
    """
    dx = 1.0
    dp = 1.0
    
    overlap = np.sum(pred_wigner * target_wigner) * dx * dp
    norm_pred = np.sum(pred_wigner**2) * dx * dp
    norm_target = np.sum(target_wigner**2) * dx * dp
    
    if norm_pred > 0 and norm_target > 0:
        fidelity = overlap / np.sqrt(norm_pred * norm_target)
        return np.clip(fidelity, 0, 1)
    
    return 0.0


def calibrate_noise_for_fidelity(ideal_wigner, target_fidelity=0.95,
                                  base_params=None, max_iterations=20):
    """
    校准噪声参数以达到目标保真度
    """
    if base_params is None:
        base_params = {
            'detection_efficiency': 0.90,
            'dark_count_rate': 0.005,
            'readout_noise_std': 0.015,
            'shot_noise_scale': 0.03,
            'calibration_drift': 0.005,
            'background_level': 0.003,
        }
    
    low_scale, high_scale = 0.1, 5.0
    best_scale = 1.0
    
    print(f"开始噪声校准，目标保真度: {target_fidelity:.3f}")
    
    for iteration in range(max_iterations):
        mid_scale = (low_scale + high_scale) / 2
        noise_model = ExperimentalNoise(**base_params, noise_scale=mid_scale)
        distorted = noise_model.apply_state_distortion(ideal_wigner.copy())
        fidelity = compute_fidelity(distorted, ideal_wigner)
        
        print(f"  迭代 {iteration+1}: scale={mid_scale:.3f}, F={fidelity:.4f}")
        
        if abs(fidelity - target_fidelity) < 0.005:
            best_scale = mid_scale
            print(f"✓ 校准完成: scale={best_scale:.3f}, F={fidelity:.4f}")
            break
        
        if fidelity > target_fidelity:
            low_scale = mid_scale
        else:
            high_scale = mid_scale
        best_scale = mid_scale
    
    final_params = base_params.copy()
    final_params['noise_scale'] = best_scale
    return ExperimentalNoise(**final_params)
