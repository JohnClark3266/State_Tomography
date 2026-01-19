"""
GKP态稀疏量子态层析 - 基于主动学习的CNN方法

目标: 用CNN从稀疏采样点重建完整Wigner函数，然后计算密度矩阵和保真度
方法: 使用5个CNN委员会进行主动学习，选择高不确定性区域进行采样

参考: active_learning_2d.py
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
# 2. 稀疏采样与掩码
# =========================

def create_sparse_input(full_wigner, sampling_mask):
    """
    从完整Wigner函数创建稀疏输入
    
    参数:
        full_wigner: 完整的Wigner函数 (H, W)
        sampling_mask: 采样掩码，True表示已测量 (H, W)
    
    返回:
        sparse_input: 稀疏输入 (2, H, W) - [采样值, 掩码]
    """
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


def generate_training_data(n_samples=150, grid_size=64, sampling_ratios=(0.05, 0.25)):
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
    
    for i in range(n_samples):
        _, _, full_wigner = create_gkp_grid(grid_size=grid_size, delta=deltas[i])
        mask = generate_random_mask(grid_size, ratios[i])
        sparse_input = create_sparse_input(full_wigner, mask)
        
        inputs.append(sparse_input)
        targets.append(full_wigner[np.newaxis, :, :])
    
    inputs = np.array(inputs)   # (N, 2, H, W)
    targets = np.array(targets) # (N, 1, H, W)
    
    return inputs, targets


# =========================
# 3. 五种CNN架构 (修改为2通道输入)
# =========================

class CNN1(nn.Module):
    """3层卷积网络，ReLU激活"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 2通道输入
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
# 4. 保真度计算
# =========================

def compute_fidelity(pred_wigner, target_wigner, dx=None, dp=None):
    """
    从Wigner函数计算保真度
    
    对于纯态，保真度可以通过Wigner函数的重叠积分近似:
    F ≈ 2π ∫∫ W_pred(x,p) * W_target(x,p) dx dp
    
    注意: 这是一个近似方法，对于纯态足够准确
    """
    if dx is None:
        dx = 8.0 / pred_wigner.shape[0]  # 假设范围是[-4, 4]
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
# 5. 主动学习稀疏层析
# =========================

class ActiveSparseTomography:
    """基于主动学习的稀疏量子态层析"""
    
    def __init__(self, grid_size=64, initial_ratio=0.05, add_ratio=0.03,
                 rounds=8, epochs=80, lr=1e-3, target_delta=0.3):
        self.grid_size = grid_size
        self.initial_ratio = initial_ratio
        self.add_ratio = add_ratio
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr
        self.target_delta = target_delta
        
        # 设备选择
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 创建目标GKP态
        self.X, self.P, self.target_wigner = create_gkp_grid(
            grid_size=grid_size, delta=target_delta
        )
        
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
        }
    
    def _generate_training_data(self, n_samples=200):
        """生成当前采样率下的训练数据"""
        current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        inputs, targets = generate_training_data(
            n_samples=n_samples,
            grid_size=self.grid_size,
            sampling_ratios=(max(0.05, current_ratio - 0.05), current_ratio + 0.05)
        )
        return inputs, targets
    
    def _train_committee(self, inputs, targets):
        """训练CNN委员会"""
        train_inputs = torch.tensor(inputs, dtype=torch.float32)
        train_targets = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 增大batch size加速
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
        # 将已采样点的方差设为-∞
        candidate_var = variance.copy()
        candidate_var[self.sampling_mask] = -np.inf
        
        # 选择方差最大的点
        flat_indices = np.argsort(candidate_var.ravel())[-n_points:]
        new_mask = np.zeros_like(self.sampling_mask)
        new_mask.flat[flat_indices] = True
        
        return new_mask
    
    def run(self):
        """运行主动学习稀疏层析"""
        print("="*60)
        print("主动学习稀疏量子态层析")
        print("="*60)
        print(f"目标: GKP态 (δ={self.target_delta})")
        print(f"网格大小: {self.grid_size}x{self.grid_size}")
        print(f"初始采样率: {self.initial_ratio*100:.1f}%")
        print("="*60 + "\n")
        
        # 初始随机采样
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = np.random.choice(
            self.grid_size ** 2, n_initial, replace=False
        )
        self.sampling_mask.flat[initial_indices] = True
        
        for round_id in range(self.rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            print(f"[Round {round_id+1}/{self.rounds}] 采样率: {current_ratio*100:.1f}%")
            
            # 生成训练数据并训练
            print("  训练CNN委员会...")
            inputs, targets = self._generate_training_data()
            self._train_committee(inputs, targets)
            
            # 用当前采样预测
            sparse_input = create_sparse_input(self.target_wigner, self.sampling_mask)
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 计算保真度和MSE
            fidelity = compute_fidelity(mean_pred, self.target_wigner)
            mse = np.mean((mean_pred - self.target_wigner) ** 2)
            
            self.history['sampling_ratio'].append(current_ratio)
            self.history['fidelity'].append(fidelity)
            self.history['mse'].append(mse)
            
            print(f"  保真度: {fidelity:.4f}")
            print(f"  MSE: {mse:.6f}")
            
            # 选择新采样点 (除了最后一轮)
            if round_id < self.rounds - 1:
                n_new = int(self.grid_size ** 2 * self.add_ratio)
                new_points = self._select_new_points(variance, n_new)
                self.sampling_mask |= new_points
                print(f"  新增 {n_new} 个采样点")
            
            print()
        
        # 最终预测
        sparse_input = create_sparse_input(self.target_wigner, self.sampling_mask)
        self.final_pred, self.final_variance, _ = self._committee_predict(sparse_input)
        self.final_fidelity = compute_fidelity(self.final_pred, self.target_wigner)
        
        print("="*60)
        print(f"最终保真度: {self.final_fidelity:.4f}")
        print(f"最终采样率: {self.sampling_mask.sum()/(self.grid_size**2)*100:.1f}%")
        print("="*60)
    
    def plot_results(self, save_dir="tomography"):
        """绘制结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 目标Wigner函数
        im0 = axes[0, 0].contourf(self.X, self.P, self.target_wigner, 
                                   levels=40, cmap='RdBu_r')
        axes[0, 0].set_title(f"Target GKP (δ={self.target_delta})")
        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("p")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 采样点分布
        axes[0, 1].contourf(self.X, self.P, self.target_wigner, 
                           levels=30, cmap='RdBu_r', alpha=0.5)
        sample_y, sample_x = np.where(self.sampling_mask)
        x_coords = self.X[0, sample_x]
        p_coords = self.P[sample_y, 0]
        axes[0, 1].scatter(x_coords, p_coords, s=3, c='cyan', alpha=0.7)
        ratio = self.sampling_mask.sum() / (self.grid_size**2) * 100
        axes[0, 1].set_title(f"Sampling Points ({ratio:.1f}%)")
        axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("p")
        
        # 3. 重建结果
        im2 = axes[0, 2].contourf(self.X, self.P, self.final_pred,
                                   levels=40, cmap='RdBu_r')
        axes[0, 2].set_title(f"Reconstruction (F={self.final_fidelity:.4f})")
        axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("p")
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. 误差图
        error = np.abs(self.final_pred - self.target_wigner)
        im3 = axes[1, 0].contourf(self.X, self.P, error, levels=40, cmap='hot')
        axes[1, 0].set_title("Absolute Error")
        axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("p")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 5. 采样率 vs 保真度
        axes[1, 1].plot([r*100 for r in self.history['sampling_ratio']], 
                        self.history['fidelity'], 'bo-', linewidth=2, markersize=8)
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
        
        plt.suptitle("Active Learning Sparse Quantum Tomography for GKP State",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sparse_tomography_results.png", dpi=300)
        print(f"\n保存结果图到 {save_dir}/sparse_tomography_results.png")
        plt.close()


def main():
    tomography = ActiveSparseTomography(
        grid_size=100,            # 100x100分辨率 (速度优化)
        initial_ratio=0.05,       # 初始5%采样
        add_ratio=0.025,          # 每轮增加2.5%
        rounds=5,                 # 5轮主动学习 (最终17.5%采样)
        epochs=40,                # 减少epoch加速
        lr=2e-3,                  # 提高学习率加速收敛
        target_delta=0.3,
    )
    
    tomography.run()
    tomography.plot_results()
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
