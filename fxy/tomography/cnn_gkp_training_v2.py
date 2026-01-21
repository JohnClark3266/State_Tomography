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

class ExperimentalNoise:
    """模拟真实量子光学实验中的各种噪声"""
    
    def __init__(self, 
                 detection_efficiency=0.85,     # 探测效率
                 dark_count_rate=0.01,          # 暗计数率
                 readout_noise_std=0.02,        # 读出噪声标准差
                 shot_noise_scale=0.05,         # 散粒噪声强度
                 calibration_drift=0.01,        # 校准漂移
                 background_level=0.005,        # 背景噪声水平
                 noise_scale=1.0):              # 全局噪声缩放因子
        
        self.eta = detection_efficiency
        self.dark_count = dark_count_rate
        self.readout_std = readout_noise_std
        self.shot_scale = shot_noise_scale
        self.calib_drift = calibration_drift
        self.bg_level = background_level
        self.noise_scale = noise_scale  # 用于整体调整噪声强度
    
    def add_noise(self, wigner_values, mask):
        """
        对测量的Wigner函数值添加实验噪声
        
        参数:
            wigner_values: 理论Wigner函数值
            mask: 采样掩码
        
        返回:
            noisy_values: 带噪声的测量值
        """
        noisy = wigner_values.copy()
        
        # 只对采样点添加噪声
        sampled_indices = np.where(mask)
        
        for i, j in zip(*sampled_indices):
            true_value = wigner_values[i, j]
            
            # 1. 探测效率损失
            measured = true_value * self.eta
            
            # 2. 散粒噪声 (泊松统计，取决于信号强度)
            # 将Wigner值转换为等效光子数再添加泊松噪声
            if measured > 0:
                signal_strength = abs(measured) / (np.max(abs(wigner_values)) + 1e-10)
                shot_noise = np.random.normal(0, self.shot_scale * np.sqrt(signal_strength))
                measured += shot_noise * self.noise_scale
            
            # 3. 读出噪声 (高斯白噪声)
            readout_noise = np.random.normal(0, self.readout_std)
            measured += readout_noise * self.noise_scale
            
            # 4. 暗计数/背景噪声
            dark_noise = np.random.exponential(self.dark_count) * np.random.choice([-1, 1])
            background = np.random.normal(self.bg_level, self.bg_level * 0.3)
            measured += (dark_noise + background) * self.noise_scale
            
            # 5. 系统校准漂移 (缓慢变化的偏移)
            # 使用位置相关的漂移模拟系统误差
            position_drift = self.calib_drift * np.sin(i * 0.1) * np.cos(j * 0.1)
            measured += position_drift * self.noise_scale
            
            noisy[i, j] = measured
        
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
        target_fidelity: 目标保真度（实验态 vs 理论态）
        base_noise_params: 基础噪声参数
        max_iterations: 最大迭代次数
    
    返回:
        校准后的噪声模型
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
    
    grid_size = ideal_wigner.shape[0]
    full_mask = np.ones_like(ideal_wigner, dtype=bool)
    
    print(f"\n开始噪声校准，目标保真度: {target_fidelity:.3f}")
    
    for iteration in range(max_iterations):
        mid_scale = (low_scale + high_scale) / 2
        
        # 创建噪声模型
        noise_model = ExperimentalNoise(**base_noise_params, noise_scale=mid_scale)
        
        # 生成带噪声的实验态
        np.random.seed(42 + iteration)  # 保证可重复性
        noisy_wigner = noise_model.add_noise(ideal_wigner.copy(), full_mask)
        
        # 计算保真度
        fidelity = compute_fidelity(noisy_wigner, ideal_wigner)
        
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
    np.random.seed(42)  # 重置随机种子
    final_noise_params = base_noise_params.copy()
    final_noise_params['noise_scale'] = best_scale
    return ExperimentalNoise(**final_noise_params)


# =========================
# 3. 稀疏采样与掩码
# =========================

def create_sparse_input(full_wigner, sampling_mask, noise_model=None):
    """
    从完整Wigner函数创建稀疏输入（可选添加噪声）
    
    参数:
        full_wigner: 完整的Wigner函数 (H, W)
        sampling_mask: 采样掩码，True表示已测量 (H, W)
        noise_model: ExperimentalNoise实例，如果为None则不添加噪声
    
    返回:
        sparse_input: 稀疏输入 (2, H, W) - [采样值, 掩码]
    """
    if noise_model is not None:
        # 模拟实验测量：只在采样点添加噪声
        noisy_wigner = noise_model.add_noise(full_wigner, sampling_mask)
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
    生成训练数据
    
    输入: 稀疏采样的Wigner值 + 掩码 (2通道)
    目标: 完整的Wigner函数 (1通道)
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
        _, _, full_wigner = create_gkp_grid(grid_size=grid_size, delta=deltas[i])
        mask = generate_random_mask(grid_size, ratios[i])
        sparse_input = create_sparse_input(full_wigner, mask, noise_model)
        
        inputs.append(sparse_input)
        targets.append(full_wigner[np.newaxis, :, :])
    
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
    """基于主动学习的稀疏量子态层析 - 实验数据模拟版本"""
    
    def __init__(self, grid_size=64, initial_ratio=0.03, add_ratio=0.015,
                 max_rounds=15, max_sampling_ratio=0.12, epochs=60, lr=1e-3, 
                 target_delta=0.3, noise_params=None, fidelity_threshold=0.995):
        self.grid_size = grid_size
        self.initial_ratio = initial_ratio
        self.add_ratio = add_ratio
        self.max_rounds = max_rounds
        self.max_sampling_ratio = max_sampling_ratio
        self.epochs = epochs
        self.lr = lr
        self.target_delta = target_delta
        self.fidelity_threshold = fidelity_threshold
        
        # 实验噪声模型
        if noise_params is not None:
            self.noise_model = ExperimentalNoise(**noise_params)
            print(f"实验噪声模型: {self.noise_model}")
        else:
            self.noise_model = None
            print("纯理论模式（无噪声）")
        
        # 设备选择
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 创建理论GKP态（用于可视化对比）
        self.X, self.P, self.ideal_wigner = create_gkp_grid(
            grid_size=grid_size, delta=target_delta
        )
        
        # 创建真实实验态（带噪声的完整Wigner函数）
        # 这是我们的ground truth - 模拟真实实验中会得到的态
        if self.noise_model is not None:
            # 对所有点添加噪声，模拟真实实验中的量子态
            full_mask = np.ones((grid_size, grid_size), dtype=bool)
            self.target_wigner = self.noise_model.add_noise(self.ideal_wigner, full_mask)
            print("\n⚠ 目标态: 带噪声的真实实验态（更符合实验实际）")
        else:
            self.target_wigner = self.ideal_wigner.copy()
            print("\n目标态: 理想无噪声GKP态")
        
        # 初始化CNN委员会
        self.models = build_cnn_committee()
        for name, model in self.models:
            model.to(self.device)
        
        # 采样掩码
        self.sampling_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # 记录
        self.history = {
            'sampling_ratio': [],
            'fidelity': [],
            'mse': [],
            'round': [],
        }
        
        # 早停标志
        self.early_stopped = False
        self.stop_round = -1
    
    def _generate_training_data(self, n_samples=120):
        """生成当前采样率下的训练数据"""
        current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        
        # 训练数据也添加噪声以提高泛化能力
        noise_params = None
        if self.noise_model is not None:
            noise_params = {
                'detection_efficiency': self.noise_model.eta,
                'dark_count_rate': self.noise_model.dark_count,
                'readout_noise_std': self.noise_model.readout_std,
                'shot_noise_scale': self.noise_model.shot_scale,
                'calibration_drift': self.noise_model.calib_drift,
                'background_level': self.noise_model.bg_level,
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
        print("主动学习稀疏量子态层析 - 实验数据模拟")
        print("="*60)
        print(f"目标: GKP态 (δ={self.target_delta})")
        print(f"网格大小: {self.grid_size}x{self.grid_size}")
        print(f"初始采样率: {self.initial_ratio*100:.1f}%")
        print(f"最大采样率: {self.max_sampling_ratio*100:.1f}%")
        print(f"保真度阈值: {self.fidelity_threshold}")
        print("="*60 + "\n")
        
        # 初始随机采样
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = np.random.choice(
            self.grid_size ** 2, n_initial, replace=False
        )
        self.sampling_mask.flat[initial_indices] = True
        
        for round_id in range(self.max_rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            
            # 检查是否超过最大采样率
            if current_ratio > self.max_sampling_ratio:
                print(f"⚠ 已达到最大采样率 {self.max_sampling_ratio*100:.1f}%，停止训练")
                break
            
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样率: {current_ratio*100:.2f}%")
            
            # 生成训练数据并训练
            print("  训练CNN委员会...")
            inputs, targets = self._generate_training_data()
            self._train_committee(inputs, targets)
            
            # 用当前采样预测（带噪声）
            sparse_input = create_sparse_input(
                self.target_wigner, self.sampling_mask, self.noise_model
            )
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 计算保真度和MSE
            fidelity = compute_fidelity(mean_pred, self.target_wigner)
            mse = np.mean((mean_pred - self.target_wigner) ** 2)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(current_ratio)
            self.history['fidelity'].append(fidelity)
            self.history['mse'].append(mse)
            
            print(f"  保真度: {fidelity:.5f}")
            print(f"  MSE: {mse:.6f}")
            
            # 早停检查
            if fidelity >= self.fidelity_threshold:
                print(f"\n✓ 保真度已达到阈值 {self.fidelity_threshold}，提前停止！")
                self.early_stopped = True
                self.stop_round = round_id + 1
                break
            
            # 选择新采样点
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
        self.final_fidelity = compute_fidelity(self.final_pred, self.target_wigner)
        self.final_ratio = self.sampling_mask.sum()/(self.grid_size**2)
        
        print("="*60)
        print(f"最终保真度: {self.final_fidelity:.5f}")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        if self.early_stopped:
            print(f"提前停止于第 {self.stop_round} 轮")
        print("="*60)
    
    def plot_results(self, save_dir="tomography"):
        """绘制结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 目标Wigner函数（真实实验态）
        im0 = axes[0, 0].contourf(self.X, self.P, self.target_wigner, 
                                   levels=40, cmap='RdBu_r')
        title_str = f"Target: Experimental State (δ={self.target_delta})"
        if self.noise_model is not None:
            title_str += "\n(with noise)"
        axes[0, 0].set_title(title_str)
        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("p")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 采样点分布
        axes[0, 1].contourf(self.X, self.P, self.target_wigner, 
                           levels=30, cmap='RdBu_r', alpha=0.5)
        sample_y, sample_x = np.where(self.sampling_mask)
        x_coords = self.X[0, sample_x]
        p_coords = self.P[sample_y, 0]
        axes[0, 1].scatter(x_coords, p_coords, s=3, c='cyan', alpha=0.7)
        ratio = self.final_ratio * 100
        title_str = f"Sampling Points ({ratio:.2f}%)"
        if self.noise_model is not None:
            title_str += f"\n(η={self.noise_model.eta:.2f})"
        axes[0, 1].set_title(title_str)
        axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("p")
        
        # 3. 重建结果
        im2 = axes[0, 2].contourf(self.X, self.P, self.final_pred,
                                   levels=40, cmap='RdBu_r')
        recon_title = f"Reconstruction (F={self.final_fidelity:.5f})"
        if self.early_stopped:
            recon_title += f"\n(Early stop @ R{self.stop_round})"
        axes[0, 2].set_title(recon_title)
        axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("p")
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. 误差图
        error = np.abs(self.final_pred - self.target_wigner)
        im3 = axes[1, 0].contourf(self.X, self.P, error, levels=40, cmap='hot')
        axes[1, 0].set_title(f"Absolute Error (max={error.max():.4f})")
        axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("p")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 5. 采样率 vs 保真度
        axes[1, 1].plot([r*100 for r in self.history['sampling_ratio']], 
                        self.history['fidelity'], 'bo-', linewidth=2, markersize=8)
        if self.fidelity_threshold < 1.0:
            axes[1, 1].axhline(y=self.fidelity_threshold, color='r', 
                              linestyle='--', label=f'Threshold={self.fidelity_threshold}')
            axes[1, 1].legend()
        axes[1, 1].set_xlabel("Sampling Ratio (%)")
        axes[1, 1].set_ylabel("Fidelity")
        axes[1, 1].set_title("Fidelity vs Sampling Ratio")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])
        
        # 6. 不确定性图
        im5 = axes[1, 2].contourf(self.X, self.P, self.final_variance,
                                   levels=40, cmap='viridis')
        axes[1, 2].set_title("Uncertainty (Variance)")
        axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("p")
        plt.colorbar(im5, ax=axes[1, 2])
        
        title_str = "Active Learning Sparse Quantum Tomography - Experimental Simulation"
        if self.noise_model is not None:
            title_str += f" (with noise)"
        plt.suptitle(title_str, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
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
        max_rounds=10,               # 最多10轮
        max_sampling_ratio=0.12,     # 最大12%采样率
        epochs=50,                   # 50个epoch
        lr=2e-3,                     # 学习率
        target_delta=0.3,            # GKP参数
        noise_params=noise_params,   # 实验噪声
        fidelity_threshold=0.995,    # 保真度阈值
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
    print(f"  最终保真度: {tomography.final_fidelity:.5f}")
    print(f"  训练轮数: {len(tomography.history['round'])}")
    if tomography.early_stopped:
        print(f"  早停: 是 (第{tomography.stop_round}轮)")
    else:
        print(f"  早停: 否")
    print("="*60)


if __name__ == "__main__":
    main()
