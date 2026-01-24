"""
猫态稀疏量子态层析 - 理论态测试 (无噪声)

用于测试理论态的层析效果。
"""

import sys
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cat_state import create_cat_grid, cat_wigner
from sparse_sampling import create_sparse_input, generate_random_mask
from cnn_models import build_cnn_committee
from fidelity import compute_fidelity

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)


def generate_cat_training_data_ideal(n_samples=150, grid_size=64, 
                                      sampling_ratios=(0.05, 0.25), x_range=(-5, 5)):
    """生成无噪声的猫态训练数据"""
    inputs = []
    targets = []
    
    alphas = np.random.uniform(1.5, 3.0, n_samples)
    parities = np.random.choice(['even', 'odd'], n_samples)
    ratios = np.random.uniform(sampling_ratios[0], sampling_ratios[1], n_samples)
    
    for i in range(n_samples):
        _, _, ideal_wigner = create_cat_grid(grid_size=grid_size, alpha=alphas[i], 
                                              parity=parities[i], x_range=x_range)
        
        target_wigner = ideal_wigner.copy()
        mask = generate_random_mask(grid_size, ratios[i])
        sparse_input = create_sparse_input(target_wigner, mask, noise_model=None)
        
        inputs.append(sparse_input)
        targets.append(target_wigner[np.newaxis, :, :])
    
    return np.array(inputs), np.array(targets)


class CatStateTomographyIdeal:
    """猫态稀疏量子态层析 - 理论态版本（无噪声）"""
    
    def __init__(self, grid_size=64, alpha=2.5, parity='even', x_range=(-5, 5),
                 initial_ratio=0.03, add_ratio=0.015, max_rounds=20, 
                 epochs=50, lr=2e-3, F2_threshold=0.99):
        
        self.grid_size = grid_size
        self.alpha = alpha
        self.parity = parity
        self.x_range = x_range
        self.initial_ratio = initial_ratio
        self.add_ratio = add_ratio
        self.max_rounds = max_rounds
        self.epochs = epochs
        self.lr = lr
        self.F2_threshold = F2_threshold
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 创建理想猫态（无噪声）
        self.X, self.P, self.target_wigner = create_cat_grid(
            grid_size=grid_size, alpha=alpha, parity=parity, x_range=x_range
        )
        
        print(f"\n目标态: {parity}猫态 (α={alpha})")
        print(f"网格大小: {grid_size}x{grid_size}")
        print(f"无噪声模式 - 理论态测试")
        
        self.models = build_cnn_committee()
        for name, model in self.models:
            model.to(self.device)
        
        self.sampling_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        self.history = {
            'sampling_ratio': [], 'fidelity': [], 'mse': [], 'round': []
        }
        
        self.early_stopped = False
        self.stop_round = -1
        self.final_pred = None
        self.final_fidelity = None
        self.final_ratio = None
    
    def _generate_training_data(self, n_samples=100):
        current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        return generate_cat_training_data_ideal(
            n_samples=n_samples,
            grid_size=self.grid_size,
            sampling_ratios=(max(0.03, current_ratio - 0.03), 
                           min(0.20, current_ratio + 0.05)),
            x_range=self.x_range
        )
    
    def _train_committee(self, inputs, targets):
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
        return predictions.mean(axis=0), predictions.var(axis=0), predictions
    
    def _select_new_points(self, variance, n_points):
        candidate_var = variance.copy()
        candidate_var[self.sampling_mask] = -np.inf
        flat_indices = np.argsort(candidate_var.ravel())[-n_points:]
        new_mask = np.zeros_like(self.sampling_mask)
        new_mask.flat[flat_indices] = True
        return new_mask
    
    def run(self):
        parity_cn = "偶" if self.parity == 'even' else "奇"
        print("\n" + "="*60)
        print(f"主动学习稀疏量子态层析 - {parity_cn}猫态 (理论态, α={self.alpha})")
        print("="*60)
        print(f"目标保真度阈值: F ≥ {self.F2_threshold}")
        print(f"最大轮数: {self.max_rounds}")
        print(f"每轮epochs: {self.epochs}")
        print("="*60 + "\n")
        
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = np.random.choice(self.grid_size ** 2, n_initial, replace=False)
        self.sampling_mask.flat[initial_indices] = True
        
        for round_id in range(self.max_rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样率: {current_ratio*100:.2f}%")
            
            print("  训练CNN委员会...")
            inputs, targets = self._generate_training_data()
            self._train_committee(inputs, targets)
            
            sparse_input = create_sparse_input(
                self.target_wigner, self.sampling_mask, noise_model=None
            )
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            fidelity = compute_fidelity(mean_pred, self.target_wigner)
            mse = np.mean((mean_pred - self.target_wigner) ** 2)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(current_ratio)
            self.history['fidelity'].append(fidelity)
            self.history['mse'].append(mse)
            
            print(f"  保真度: {fidelity:.5f}")
            print(f"  MSE: {mse:.6f}")
            
            if fidelity >= self.F2_threshold:
                print(f"\n✓ 保真度已达到阈值 {self.F2_threshold}，提前停止！")
                self.early_stopped = True
                self.stop_round = round_id + 1
                break
            
            if round_id < self.max_rounds - 1:
                n_new = int(self.grid_size ** 2 * self.add_ratio)
                new_points = self._select_new_points(variance, n_new)
                self.sampling_mask |= new_points
                print(f"  新增 {n_new} 个采样点\n")
        
        sparse_input = create_sparse_input(
            self.target_wigner, self.sampling_mask, noise_model=None
        )
        self.final_pred, self.final_variance, _ = self._committee_predict(sparse_input)
        self.final_fidelity = compute_fidelity(self.final_pred, self.target_wigner)
        self.final_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        
        print("\n" + "="*60)
        print("实验结束总结:")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        print(f"最终保真度: {self.final_fidelity:.5f}")
        if self.early_stopped:
            print(f"提前停止于第 {self.stop_round} 轮")
        print("="*60)
    
    def plot_results(self, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)
        parity_cn = "偶" if self.parity == 'even' else "奇"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标态
        im0 = axes[0, 0].contourf(self.X, self.P, self.target_wigner, levels=40, cmap='RdBu_r')
        axes[0, 0].set_title(f"Target: {parity_cn}Cat State (α={self.alpha})")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 重建态
        im1 = axes[0, 1].contourf(self.X, self.P, self.final_pred, levels=40, cmap='RdBu_r')
        axes[0, 1].set_title(f"Reconstruction (F={self.final_fidelity:.4f})")
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 误差
        error = self.final_pred - self.target_wigner
        im2 = axes[0, 2].contourf(self.X, self.P, error, levels=40, cmap='RdBu_r')
        axes[0, 2].set_title(f"Error (Max={np.max(np.abs(error)):.4f})")
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. 采样分布
        axes[1, 0].contourf(self.X, self.P, self.target_wigner, levels=30, cmap='RdBu_r', alpha=0.5)
        sample_y, sample_x = np.where(self.sampling_mask)
        axes[1, 0].scatter(self.X[0, sample_x], self.P[sample_y, 0], s=3, c='cyan', alpha=0.7)
        axes[1, 0].set_title(f"Sampling Points ({self.final_ratio*100:.2f}%)")
        
        # 5. 保真度曲线
        ax5 = axes[1, 1]
        x_vals = [r*100 for r in self.history['sampling_ratio']]
        ax5.plot(x_vals, self.history['fidelity'], 'b-o', label='Fidelity')
        ax5.axhline(y=self.F2_threshold, color='r', linestyle=':', label='Target')
        ax5.set_xlabel("Sampling %")
        ax5.set_ylabel("Fidelity")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title("Fidelity vs Sampling Rate")
        
        # 6. 不确定性
        im5 = axes[1, 2].contourf(self.X, self.P, self.final_variance, levels=40, cmap='viridis')
        axes[1, 2].set_title("Uncertainty Map")
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.suptitle(f"Ideal {parity_cn}Cat State Tomography (α={self.alpha}, epochs={self.epochs})", 
                     fontsize=14, fontweight='bold')
        
        filename = f"cat_ideal_{self.parity}_alpha{self.alpha}.png"
        filepath = f"{save_dir}/{filename}"
        plt.savefig(filepath, dpi=300)
        print(f"\n保存结果图到 {filepath}")
        plt.close()


def main():
    """主函数 - 理论态测试"""
    
    print("\n" + "="*70)
    print("       偶猫态 (Even Cat State) 理论态层析测试")
    print("="*70)
    
    tomo = CatStateTomographyIdeal(
        grid_size=64,
        alpha=2.5,
        parity='even',
        x_range=(-5, 5),
        initial_ratio=0.03,
        add_ratio=0.015,
        max_rounds=20,
        epochs=50,               # 50 epochs
        lr=2e-3,
        F2_threshold=0.995       # 理论态应该能达到更高保真度
    )
    
    tomo.run()
    tomo.plot_results()
    
    print("\n程序执行完成!")
    print(f"\n最终结果:")
    print(f"  保真度: {tomo.final_fidelity:.5f}")
    print(f"  采样率: {tomo.final_ratio*100:.2f}%")


if __name__ == "__main__":
    main()
