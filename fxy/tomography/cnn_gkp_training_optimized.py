"""
GKP态稀疏量子态层析 - 基于主动学习的CNN方法 (优选委员会版)

目标: 用CNN从稀疏采样点重建完整Wigner函数，然后计算密度矩阵和保真度
改进: 在主动学习开始前，从候选池中根据理论值(Target)筛选最佳的委员会成员
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
# 3. 五种CNN架构 (与原版一致)
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


def build_candidate_pool(n_per_type=3):
    """构建更大的候选模型池 (每种类型n_per_type个实例)"""
    candidates = []
    
    # 基础架构列表
    architectures = [
        ("CNN1_ReLU", CNN1),
        ("CNN2_Tanh", CNN2),
        ("CNN3_LeakyReLU", CNN3),
        ("CNN4_ResNet", CNN4),
        ("CNN5_UNet", CNN5),
    ]
    
    for base_name, ModelClass in architectures:
        for i in range(n_per_type):
            name = f"{base_name}_{i}"
            # 每次实例化ModelClass()都会有新的随机初始化权重
            candidates.append((name, ModelClass()))
            
    return candidates


# =========================
# 4. 保真度计算
# =========================

def compute_fidelity(pred_wigner, target_wigner, dx=None, dp=None):
    """
    从Wigner函数计算保真度
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
# 5. 主动学习稀疏层析 (优选委员会版)
# =========================

class ActiveSparseTomography:
    """基于主动学习的稀疏量子态层析"""
    
    def __init__(self, grid_size=64, initial_ratio=0.05, add_ratio=0.03,
                 rounds=8, epochs=80, lr=1e-3, target_delta=0.3, 
                 n_candidates_per_type=3, committee_size=5):
        self.grid_size = grid_size
        self.initial_ratio = initial_ratio
        self.add_ratio = add_ratio
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr
        self.target_delta = target_delta
        self.committee_size = committee_size
        
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
        
        # 构建候选池
        self.candidates = build_candidate_pool(n_per_type=n_candidates_per_type)
        print(f"构建候选模型池: {len(self.candidates)} 个模型")
        for name, model in self.candidates:
            model.to(self.device)
        
        # 最终选定的委员会
        self.models = [] 
        
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
        # 保证采样的比例范围合理
        min_r = max(0.01, current_ratio - 0.05)
        max_r = min(1.0, current_ratio + 0.1)
        
        inputs, targets = generate_training_data(
            n_samples=n_samples,
            grid_size=self.grid_size,
            sampling_ratios=(min_r, max_r)
        )
        return inputs, targets
    
    def _train_models(self, models_list, inputs, targets, specific_epochs=None):
        """训练指定的模型列表"""
        train_inputs = torch.tensor(inputs, dtype=torch.float32)
        train_targets = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(train_inputs, train_targets)
        # 增大 batch size 加速
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        
        epochs_to_run = specific_epochs if specific_epochs is not None else self.epochs
        
        for name, model in models_list:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            for epoch in range(epochs_to_run):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
    
    def _select_best_committee(self):
        """
        在初始阶段评估所有候选模型，选择表现最好的前K个
        利用 'Target Truth' 也就是 self.target_wigner (理论值)
        """
        print(f"\n开始委员会筛选 (候选数: {len(self.candidates)}, 目标选: {self.committee_size})...")
        print("1. 生成筛选用训练数据...")
        # 生成一批训练数据用于初步训练 (基于初始采样mask)
        train_inputs, train_targets = self._generate_training_data(n_samples=200)
        
        # 2. 训练所有候选模型
        print("2. 训练所有候选模型...")
        # 预选阶段可以少训练几个epoch以节省时间，或者保持一致
        self._train_models(self.candidates, train_inputs, train_targets, specific_epochs=30)
        
        # 3. 在目标态上评估表现
        print("3. 基于理论值 (Target Truth) 评估模型...")
        scores = []
        
        sparse_input = create_sparse_input(self.target_wigner, self.sampling_mask)
        input_tensor = torch.tensor(
            sparse_input[np.newaxis, :, :, :], dtype=torch.float32
        ).to(self.device)
        
        for name, model in self.candidates:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            
            # 计算保真度作为得分
            fid = compute_fidelity(pred, self.target_wigner)
            mse = np.mean((pred - self.target_wigner) ** 2)
            scores.append((name, model, fid, mse))
        
        # 4. 排序并选择
        # 按保真度降序排序
        scores.sort(key=lambda x: x[2], reverse=True)
        
        print("\n候选模型评估结果:")
        print(f"{'Rank':<5} {'Model Name':<20} {'Fidelity':<10} {'MSE':<10}")
        print("-" * 50)
        for i, (name, _, fid, mse) in enumerate(scores):
            mark = "*" if i < self.committee_size else " "
            print(f"{mark}{i+1:<4} {name:<20} {fid:.4f}     {mse:.6f}")
            
        # 选出前 K 个
        self.models = [(name, model) for name, model, _, _ in scores[:self.committee_size]]
        print(f"\n已选定 {len(self.models)} 个最佳模型作为委员会成员。")
        print("-" * 50)

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
        print("主动学习稀疏量子态层析 - 优选委员会版")
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
        
        # ==========================================
        # 新增: 委员会成员筛选阶段
        # ==========================================
        self._select_best_committee()
        
        # 开始主动学习循环
        for round_id in range(self.rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            print(f"\n[Round {round_id+1}/{self.rounds}] 采样率: {current_ratio*100:.1f}%")
            
            # 生成训练数据并训练 (只训练选定的委员会)
            print("  训练委员会成员...")
            inputs, targets = self._generate_training_data()
            self._train_models(self.models, inputs, targets)
            
            # 用当前采样预测
            sparse_input = create_sparse_input(self.target_wigner, self.sampling_mask)
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 计算保真度和MSE
            fidelity = compute_fidelity(mean_pred, self.target_wigner)
            mse = np.mean((mean_pred - self.target_wigner) ** 2)
            
            self.history['sampling_ratio'].append(current_ratio)
            self.history['fidelity'].append(fidelity)
            self.history['mse'].append(mse)
            
            print(f"  当前保真度: {fidelity:.4f}")
            print(f"  当前MSE: {mse:.6f}")
            
            # 选择新采样点 (除了最后一轮)
            if round_id < self.rounds - 1:
                n_new = int(self.grid_size ** 2 * self.add_ratio)
                new_points = self._select_new_points(variance, n_new)
                self.sampling_mask |= new_points
                print(f"  新增 {n_new} 个采样点 (基于最大方差)")
            
        # 最终预测
        sparse_input = create_sparse_input(self.target_wigner, self.sampling_mask)
        self.final_pred, self.final_variance, _ = self._committee_predict(sparse_input)
        self.final_fidelity = compute_fidelity(self.final_pred, self.target_wigner)
        
        print("\n" + "="*60)
        print(f"最终保真度: {self.final_fidelity:.4f}")
        print(f"最终采样率: {self.sampling_mask.sum()/(self.grid_size**2)*100:.1f}%")
        print("="*60)
    
    def plot_results(self, save_dir="tomography"):
        """绘制结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10)) # 稍微调宽一点
        
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
        
        plt.suptitle("Optimized Active Learning: Pre-selected Committee",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/optimized_tomography_results.png", dpi=300)
        print(f"\n保存结果图到 {save_dir}/optimized_tomography_results.png")
        plt.close()


def main():
    tomography = ActiveSparseTomography(
        grid_size=100,            # 100x100分辨率
        initial_ratio=0.05,       # 初始5%采样
        add_ratio=0.025,          # 每轮增加2.5%
        rounds=5,                 # 5轮主动学习
        epochs=40,                # 40个epoch
        lr=2e-3,                  # 学习率
        target_delta=0.3,
        n_candidates_per_type=3,  # 每种架构生成3个候选 (共15个)
        committee_size=5          # 最终选5个进委员会
    )
    
    tomography.run()
    tomography.plot_results()
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
