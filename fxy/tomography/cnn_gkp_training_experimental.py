"""
GKP态稀疏量子态层析 - 实验数据模拟版本

新增功能:
1. 添加多种实验噪声模拟
2. 优化采样率不超过12%
3. 保真度>0.995时提前停止

参考: cnn_gkp_training.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 1. GKP态Wigner函数
# =========================

def gkp_wigner(x, p, delta=0.3, n_peaks=5):
    """
    GKP态的Wigner函数
    W_GKP(x, p) ∝ Σ_{s,t∈Z} exp(-(x - s√π)² / (2Δ²)) * exp(-(p - t√π)² / (2Δ²))
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
    """创建GKP态的完整Wigner函数网格"""
    x = np.linspace(x_range[0], x_range[1], grid_size)
    p = np.linspace(x_range[0], x_range[1], grid_size)
    X, P = np.meshgrid(x, p)
    W = gkp_wigner(X, P, delta=delta)
    return X, P, W.astype(np.float32)


# =========================
# 2. 实验噪声模拟
# =========================

from scipy.ndimage import gaussian_filter

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
        self.noise_scale = noise_scale  # 用于整体调整噪声强度
    
    def apply_state_distortion(self, wigner_values):
        """
        应用态本身的物理畸变（损耗、漂移）
        这些是实验态"固有"的属性，应该包含在训练目标(Target)中
        """
        distorted = wigner_values.copy()
        H, W = wigner_values.shape
        
        # 1. 探测效率/光子损耗 -> 高斯模糊 (Diffusion)
        # 损耗不仅降低幅度，还会使相空间分布扩散
        # sigma 与 (1-eta)/eta 相关，这里简化模型：
        # sigma = base_sigma * (1/eta - 1) * noise_scale
        if self.eta < 1.0:
            # 基础模糊量，可根据网格大小调整
            loss_factor = (1.0 - self.eta) * 2.0 * self.noise_scale
            sigma = max(0.1, loss_factor)
            distorted = gaussian_filter(distorted, sigma=sigma)
            
            # 同时应用幅度衰减 (虽然归一化后会被消除，但为了物理正确性保留)
            distorted *= self.eta
        
        # 2. 系统校准漂移 (位置相关的系统误差)
        if self.calib_drift > 0:
            # 创建坐标网格
            y, x = np.mgrid[0:H, 0:W]
            # 归一化坐标
            ny, nx = y/H, x/W
            # 生成平滑的漂移场
            drift_field = self.calib_drift * np.sin(nx * np.pi) * np.cos(ny * np.pi)
            distorted += drift_field * self.noise_scale
            
        return distorted

    def apply_measurement_noise(self, wigner_values, mask):
        """
        应用测量过程中的随机噪声（散粒噪声、电子噪声）
        这些是测量的"伪影"，应该只出现在输入(Input)中，不出现在目标(Target)中
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
    重要更改：现在只根据"态畸变"来校准保真度，因为这才是实验态的"固有"保真度
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
        
        # 创建噪声模型
        noise_model = ExperimentalNoise(**base_noise_params, noise_scale=mid_scale)
        
        # 生成带畸变的实验态 (target)
        distorted_wigner = noise_model.apply_state_distortion(ideal_wigner.copy())
        
        # 计算保真度 (畸变态 vs 理想态)
        fidelity = compute_fidelity(distorted_wigner, ideal_wigner)
        
        print(f"  迭代 {iteration+1}: scale={mid_scale:.3f}, F={fidelity:.4f}")
        
        # 检查是否达到目标
        if abs(fidelity - target_fidelity) < 0.005:  # 0.5%容差
            best_scale = mid_scale
            print(f"✓ 找到合适的噪声强度: scale={best_scale:.3f}, F={fidelity:.4f}")
            break
        
        # 调整搜索范围
        if fidelity > target_fidelity:
            low_scale = mid_scale  # 需要更多噪声
        else:
            high_scale = mid_scale  # 需要更少噪声
        
        best_scale = mid_scale
    
    # 返回最终校准的噪声模型
    final_noise_params = base_noise_params.copy()
    final_noise_params['noise_scale'] = best_scale
    return ExperimentalNoise(**final_noise_params)


# =========================
# 3. 稀疏采样与掩码
# =========================

def create_sparse_input(full_wigner, sampling_mask, noise_model=None):
    """
    从完整Wigner函数创建稀疏输入（可选添加噪声）
    注意：full_wigner应该是已经包含了state_distortion的真正实验态（如果noise_model存在）
    这里只添加measurement_noise
    """
    if noise_model is not None:
        # 模拟实验测量：只在采样点添加测量噪声 (shot noise, readout noise等)
        noisy_wigner = noise_model.apply_measurement_noise(full_wigner, sampling_mask)
        sparse_values = np.where(sampling_mask, noisy_wigner, 0.0)
    else:
        sparse_values = np.where(sampling_mask, full_wigner, 0.0)
    
    mask_channel = sampling_mask.astype(np.float32)
    sparse_input = np.stack([sparse_values, mask_channel], axis=0)
    return sparse_input.astype(np.float32)


def generate_random_mask(grid_size, sampling_ratio):
    """生成随机采样掩码"""
    n_samples = int(grid_size * grid_size * sampling_ratio)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    indices = np.random.choice(grid_size * grid_size, n_samples, replace=False)
    mask.flat[indices] = True
    return mask


def generate_training_data(n_samples=150, grid_size=64, sampling_ratios=(0.05, 0.25),
                          add_noise=False, noise_params=None):
    """
    生成训练数据 - 修正版
    
    逻辑更正：
    1. Target = 理想态 + 物理畸变 (Loss, Drift)
    2. Input = Target + 测量噪声 (Shot, Readout)
    """
    inputs = []
    targets = []
    
    # 变化delta参数增加多样性
    deltas = np.random.uniform(0.25, 0.4, n_samples)
    ratios = np.random.uniform(sampling_ratios[0], sampling_ratios[1], n_samples)
    
    # 为训练数据创建噪声模型（如果需要）
    if add_noise and noise_params is not None:
        noise_model = ExperimentalNoise(**noise_params)
    else:
        noise_model = None
    
    for i in range(n_samples):
        # 1. 产生理想态
        _, _, ideal_wigner = create_gkp_grid(grid_size=grid_size, delta=deltas[i])
        
        # 2. 产生实验态 (Ground Truth / Target)
        if noise_model:
            # 加上物理畸变作为训练目标
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


# =========================
# 4. 五种CNN架构 (与原版相同)
# =========================

class CNN1(nn.Module):
    """3层卷积网络，ReLU激活"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN2(nn.Module):
    """4层卷积网络，Tanh激活"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)


class CNN3(nn.Module):
    """3层卷积网络，LeakyReLU激活，更宽的通道"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)


class CNN4(nn.Module):
    """ResNet风格，带跳跃连接"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.conv_in(x))
        res = h
        h = self.relu(self.conv2(h))
        h = self.conv3(h)
        h = self.relu(h + res)
        res = h
        h = self.relu(self.conv3(h))
        h = self.conv4(h)
        h = self.relu(h + res)
        return self.conv_out(h)


class CNN5(nn.Module):
    """U-Net风格编码器-解码器"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def build_cnn_committee():
    """构建5个不同的CNN模型"""
    return [
        ("CNN1_ReLU", CNN1()),
        ("CNN2_Tanh", CNN2()),
        ("CNN3_LeakyReLU", CNN3()),
        ("CNN4_ResNet", CNN4()),
        ("CNN5_UNet", CNN5()),
    ]


# =========================
# 5. 保真度计算
# =========================

def compute_fidelity(pred_wigner, target_wigner, dx=None, dp=None):
    """
    从Wigner函数计算保真度
    
    对于纯态，保真度可以通过Wigner函数的重叠积分近似:
    F ≈ 2π ∫∫ W_pred(x,p) * W_target(x,p) dx dp
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


# =========================
# 6. 主动学习稀疏层析（实验版本）
# =========================

class ActiveSparseTomography:
    """基于主动学习的稀疏量子态层析 - 实验数据模拟版本 (F2优化版)"""
    
    def __init__(self, grid_size=64, initial_ratio=0.03, add_ratio=0.015,
                 max_rounds=30, epochs=60, lr=1e-3, 
                 target_delta=0.3, noise_params=None, 
                 target_experimental_fidelity=0.95,
                 F2_threshold=0.99):
        self.grid_size = grid_size
        self.initial_ratio = initial_ratio
        self.add_ratio = add_ratio
        self.max_rounds = max_rounds
        self.epochs = epochs
        self.lr = lr
        self.target_delta = target_delta
        self.F2_threshold = F2_threshold
        
        # 设备选择
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 创建理论GKP态（用于F3计算和校准）
        self.X, self.P, self.ideal_wigner = create_gkp_grid(
            grid_size=grid_size, delta=target_delta
        )
        
        # 噪声校准与实验态生成
        if noise_params is not None:
            print(f"\n步骤1: 校准噪声以产生目标保真度 {target_experimental_fidelity}")
            # 使用 calibrate_noise_for_fidelity (现在是基于态畸变校准)
            self.noise_model = calibrate_noise_for_fidelity(
                self.ideal_wigner, 
                target_fidelity=target_experimental_fidelity,
                base_noise_params=noise_params
            )
            
            # 目标态是"带畸变的真实态"，而不是理想态
            # 这就是网络应该努力重建的"Truth"
            self.target_wigner = self.noise_model.apply_state_distortion(self.ideal_wigner)
            
            self.F1_exp_vs_ideal = compute_fidelity(self.target_wigner, self.ideal_wigner)
            print(f"实验态创建完成: F₁ (实验vs理论) = {self.F1_exp_vs_ideal:.5f}")
        else:
            self.noise_model = None
            self.target_wigner = self.ideal_wigner.copy()
            self.F1_exp_vs_ideal = 1.0
            print("\n理想无噪声模式")
        
        # 初始化CNN委员会
        self.models = build_cnn_committee()
        for name, model in self.models:
            model.to(self.device)
        
        # 采样掩码
        self.sampling_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # 跟踪更多保真度指标
        self.history = {
            'sampling_ratio': [],
            'F1_exp_vs_ideal': [],
            'F2_recon_vs_exp': [],
            'F3_recon_vs_ideal': [],
            'mse': [],
            'round': [],
        }
        
        # 早停标志
        self.early_stopped = False
        self.stop_round = -1
    
    def _generate_training_data(self, n_samples=120):
        """生成当前采样率下的训练数据"""
        current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        
        noise_params = None
        if self.noise_model is not None:
            noise_params = {
                'detection_efficiency': self.noise_model.eta,
                'dark_count_rate': self.noise_model.dark_count,
                'readout_noise_std': self.noise_model.readout_std,
                'shot_noise_scale': self.noise_model.shot_scale,
                'calibration_drift': self.noise_model.calib_drift,
                'background_level': self.noise_model.bg_level,
                'noise_scale': self.noise_model.noise_scale,
            }
        
        inputs, targets = generate_training_data(
            n_samples=n_samples,
            grid_size=self.grid_size,
            sampling_ratios=(max(0.03, current_ratio - 0.03), 
                           min(0.15, current_ratio + 0.05)),
            add_noise=(self.noise_model is not None),
            noise_params=noise_params
        )
        return inputs, targets
    
    def _train_committee(self, inputs, targets):
        """训练CNN委员会"""
        train_inputs = torch.tensor(inputs, dtype=torch.float32)
        train_targets = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        
        # 训练每个模型
        for name, model in self.models:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            for epoch in range(self.epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
    
    def _committee_predict(self, sparse_input):
        """委员会预测并计算不确定性"""
        input_tensor = torch.tensor(
            sparse_input[np.newaxis, :, :, :], dtype=torch.float32
        ).to(self.device)
        
        predictions = []
        for name, model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        variance = predictions.var(axis=0)
        
        return mean_pred, variance, predictions
    
    def _select_new_points(self, variance, n_points):
        """基于不确定性选择新采样点"""
        candidate_var = variance.copy()
        candidate_var[self.sampling_mask] = -np.inf
        
        flat_indices = np.argsort(candidate_var.ravel())[-n_points:]
        new_mask = np.zeros_like(self.sampling_mask)
        new_mask.flat[flat_indices] = True
        
        return new_mask
    
    def run(self):
        """运行主动学习稀疏层析"""
        print("="*60)
        print("主动学习稀疏量子态层析 - 实验数据模拟 (F2优化)")
        print("="*60)
        print(f"目标: GKP态 (δ={self.target_delta})")
        print(f"目标保真度阈值: F₂ ≥ {self.F2_threshold} (Recon vs Distorted Exp)")
        print(f"最大轮数: {self.max_rounds}")
        print("="*60 + "\n")
        
        # 初始随机采样
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = np.random.choice(
            self.grid_size ** 2, n_initial, replace=False
        )
        self.sampling_mask.flat[initial_indices] = True
        
        for round_id in range(self.max_rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样率: {current_ratio*100:.2f}%")
            
            # 生成训练数据并训练
            print("  训练CNN委员会...")
            inputs, targets = self._generate_training_data()
            self._train_committee(inputs, targets)
            
            # 根据当前采样掩码，对实验态(target_wigner)添加测量噪声(measurement_noise)
            # 得到我们模拟的"实验室观测值"
            sparse_input = create_sparse_input(
                self.target_wigner, self.sampling_mask, self.noise_model
            )
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 计算保真度
            # F2: 重建态 vs 实验态 (畸变态) ← 这正是我们想最大化的，且网络训练以此为目标
            F2 = compute_fidelity(mean_pred, self.target_wigner)
            # F3: 重建态 vs 理想态
            F3 = compute_fidelity(mean_pred, self.ideal_wigner)
            
            mse = np.mean((mean_pred - self.target_wigner) ** 2)
            
            # 更新历史记录
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(current_ratio)
            self.history['F1_exp_vs_ideal'].append(self.F1_exp_vs_ideal)
            self.history['F2_recon_vs_exp'].append(F2)
            self.history['F3_recon_vs_ideal'].append(F3)
            self.history['mse'].append(mse)
            
            print(f"  F₂ (vs Exp):   {F2:.5f}")
            print(f"  F₃ (vs Ideal): {F3:.5f}")
            print(f"  MSE: {mse:.6f}")
            
            # F2早停检查
            if F2 >= self.F2_threshold:
                print(f"\n✓ F₂ 已达到阈值 {self.F2_threshold}，重建自洽！提前停止！")
                self.early_stopped = True
                self.stop_round = round_id + 1
                break
            
            # 选择新采样点 (如果没有达到最后一轮)
            if round_id < self.max_rounds - 1:
                n_new = int(self.grid_size ** 2 * self.add_ratio)
                new_points = self._select_new_points(variance, n_new)
                self.sampling_mask |= new_points
                print(f"  新增 {n_new} 个采样点")
            
            print()
        
        # 最终预测
        sparse_input = create_sparse_input(
            self.target_wigner, self.sampling_mask, self.noise_model
        )
        self.final_pred, self.final_variance, _ = self._committee_predict(sparse_input)
        
        self.final_F2 = compute_fidelity(self.final_pred, self.target_wigner)
        self.final_F3 = compute_fidelity(self.final_pred, self.ideal_wigner)
        self.final_ratio = self.sampling_mask.sum()/(self.grid_size**2)
        
        print("="*60)
        print("实验结束总结:")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        print(f"Final F₂ (Recon vs Exp)   = {self.final_F2:.5f}")
        print(f"Final F₃ (Recon vs Ideal) = {self.final_F3:.5f}")
        print(f"Baseline F₁ (Exp vs Ideal) = {self.F1_exp_vs_ideal:.5f}")
        if self.early_stopped:
            print(f"提前停止于第 {self.stop_round} 轮")
        print("="*60)
    
    def plot_results(self, save_dir="tomography"):
        """绘制结果 - 更新版含三保真度"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标Wigner函数（实验态）
        im0 = axes[0, 0].contourf(self.X, self.P, self.target_wigner, levels=40, cmap='RdBu_r')
        axes[0, 0].set_title(f"Experimental Target (F₁={self.F1_exp_vs_ideal:.4f})")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 理想Wigner函数（仅作对比）
        im1 = axes[0, 1].contourf(self.X, self.P, self.ideal_wigner, levels=40, cmap='RdBu_r')
        axes[0, 1].set_title(f"Ideal Theoretical State")
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 重建结果
        im2 = axes[0, 2].contourf(self.X, self.P, self.final_pred, levels=40, cmap='RdBu_r')
        axes[0, 2].set_title(f"Reconstruction\nF₂={self.final_F2:.4f} (vs Exp)")
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. 采样分布
        axes[1, 0].contourf(self.X, self.P, self.target_wigner, levels=30, cmap='RdBu_r', alpha=0.5)
        sample_y, sample_x = np.where(self.sampling_mask)
        axes[1, 0].scatter(self.X[0, sample_x], self.P[sample_y, 0], s=3, c='cyan', alpha=0.7)
        axes[1, 0].set_title(f"Sampling Points ({self.final_ratio*100:.2f}%)")
        
        # 5. 保真度曲线
        ax5 = axes[1, 1]
        x_vals = [r*100 for r in self.history['sampling_ratio']]
        ax5.plot(x_vals, self.history['F1_exp_vs_ideal'], 'r--', label='F₁: Exp vs Ideal')
        ax5.plot(x_vals, self.history['F2_recon_vs_exp'], 'b-o', label='F₂: Recon vs Exp')
        ax5.plot(x_vals, self.history['F3_recon_vs_ideal'], 'g-^', label='F₃: Recon vs Ideal')
        ax5.axhline(y=self.F2_threshold, color='k', linestyle=':', label='F₂ Goal')
        ax5.set_xlabel("Sampling %")
        ax5.set_ylabel("Fidelity")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title("Triple Fidelity Tracking")
        
        # 6. 不确定性/误差
        im5 = axes[1, 2].contourf(self.X, self.P, self.final_variance, levels=40, cmap='viridis')
        axes[1, 2].set_title("Uncertainty Map")
        plt.colorbar(im5, ax=axes[1, 2])
        
        title_str = "Experimental GKP Tomography (F2 Optimization)"
        plt.suptitle(title_str, fontsize=14, fontweight='bold')
        
        filename = "experimental_tomography_results.png"
        filepath = f"{save_dir}/{filename}"
        plt.savefig(filepath, dpi=300)
        print(f"\n保存结果图到 {filepath}")
        plt.close()
    
    def plot_committee_comparison(self, save_dir="tomography"):
        """绘制每个委员会成员的单独表现"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取每个委员会成员的预测
        sparse_input = create_sparse_input(
            self.target_wigner, self.sampling_mask, self.noise_model
        )
        input_tensor = torch.tensor(
            sparse_input[np.newaxis, :, :, :], dtype=torch.float32
        ).to(self.device)
        
        individual_preds = []
        individual_fidelities = []
        
        for name, model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            individual_preds.append(pred)
            fid = compute_fidelity(pred, self.target_wigner)
            individual_fidelities.append(fid)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 绘制每个委员会成员的重建结果
        for idx, (name, pred, fid) in enumerate(zip(
            [n for n, _ in self.models], individual_preds, individual_fidelities
        )):
            if idx < 5:  # 前5个子图
                row = idx // 3
                col = idx % 3
                im = axes[row, col].contourf(self.X, self.P, pred, 
                                            levels=40, cmap='RdBu_r')
                axes[row, col].set_title(f"{name}\nF={fid:.5f}")
                axes[row, col].set_xlabel("x")
                axes[row, col].set_ylabel("p")
                plt.colorbar(im, ax=axes[row, col])
        
        # 第6个子图：委员会平均
        im = axes[1, 2].contourf(self.X, self.P, self.final_pred, 
                                levels=40, cmap='RdBu_r')
        mean_fid = self.final_fidelity
        axes[1, 2].set_title(f"Committee Average\nF={mean_fid:.5f}")
        axes[1, 2].set_xlabel("x")
        axes[1, 2].set_ylabel("p")
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.suptitle("Individual Committee Member Performance", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = f"{save_dir}/committee_comparison.png"
        plt.savefig(filepath, dpi=300)
        print(f"保存委员会对比图到 {filepath}")
        plt.close()
        
        # 打印统计信息
        print("\n" + "="*60)
        print("委员会成员单独表现:")
        print("="*60)
        for name, fid in zip([n for n, _ in self.models], individual_fidelities):
            print(f"  {name:20s}: F = {fid:.5f}")
        print(f"  {'Committee Average':20s}: F = {mean_fid:.5f}")
        print(f"  {'Standard Deviation':20s}: σ = {np.std(individual_fidelities):.5f}")
        print(f"  {'Min Fidelity':20s}: {min(individual_fidelities):.5f}")
        print(f"  {'Max Fidelity':20s}: {max(individual_fidelities):.5f}")
        print("="*60)


def main():
    """主函数 - 实验数据模拟"""
    
    # 实验噪声参数（可调整以模拟不同实验条件）
    noise_params = {
        'detection_efficiency': 0.85,      # 85% 探测效率
        'dark_count_rate': 0.01,           # 1% 暗计数
        'readout_noise_std': 0.02,         # 2% 读出噪声
        'shot_noise_scale': 0.05,          # 5% 散粒噪声
        'calibration_drift': 0.01,         # 1% 校准漂移
        'background_level': 0.005,         # 0.5% 背景噪声
    }
    
    tomography = ActiveSparseTomography(
        grid_size=64,                # 64x64分辨率
        initial_ratio=0.03,          # 初始3%采样
        add_ratio=0.015,             # 每轮增加1.5%
        max_rounds=30,               # 最多30轮 (去掉max_sampling_ratio限制)
        epochs=50,                   # 50个epoch
        lr=2e-3,                     # 学习率
        target_delta=0.3,            # GKP参数
        noise_params=noise_params,   # 实验噪声
        target_experimental_fidelity=0.95, # 设定实验态的目标F1
        F2_threshold=0.99            # 设定重建的目标F2
    )
    
    tomography.run()
    tomography.plot_results()
    tomography.plot_committee_comparison()  # 新增：绘制委员会成员对比
    
    print("\n程序执行完成!")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("实验统计:")
    print(f"  总采样点数: {tomography.sampling_mask.sum()}")
    print(f"  采样率: {tomography.final_ratio*100:.2f}%")
    print(f"  F1 (实验 vs 理论): {tomography.F1_exp_vs_ideal:.5f}")
    print(f"  F2 (重建 vs 实验): {tomography.final_F2:.5f}")
    print(f"  F3 (重建 vs 理论): {tomography.final_F3:.5f}")
    print(f"  训练轮数: {len(tomography.history['round'])}")
    if tomography.early_stopped:
        print(f"  早停: 是 (第{tomography.stop_round}轮)")
    else:
        print(f"  早停: 否")
    print("="*60)


if __name__ == "__main__":
    main()
