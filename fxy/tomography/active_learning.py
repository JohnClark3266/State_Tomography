"""
主动学习稀疏量子态层析模块

提供基于主动学习的稀疏量子态层析核心类。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from .gkp_state import create_gkp_grid
from .noise_model import ExperimentalNoise, calibrate_noise_for_fidelity
from .sparse_sampling import create_sparse_input, generate_training_data
from .cnn_models import build_cnn_committee
from .fidelity import compute_fidelity


class ActiveSparseTomography:
    """基于主动学习的稀疏量子态层析 - 实验数据模拟版本 (F2优化版)"""
    
    def __init__(self, grid_size=64, initial_ratio=0.03, add_ratio=0.015,
                 max_rounds=30, epochs=60, lr=1e-3, 
                 target_delta=0.3, noise_params=None, 
                 target_experimental_fidelity=0.95,
                 F2_threshold=0.99):
        """
        初始化层析系统
        
        参数:
            grid_size: 网格尺寸
            initial_ratio: 初始采样率
            add_ratio: 每轮添加的采样率
            max_rounds: 最大训练轮数
            epochs: 每轮训练的epoch数
            lr: 学习率
            target_delta: GKP态参数
            noise_params: 噪声参数字典
            target_experimental_fidelity: 目标实验保真度 (F1)
            F2_threshold: F2早停阈值
        """
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
        
        # 创建理论GKP态
        self.X, self.P, self.ideal_wigner = create_gkp_grid(
            grid_size=grid_size, delta=target_delta
        )
        
        # 噪声校准与实验态生成
        if noise_params is not None:
            print(f"\n步骤1: 校准噪声以产生目标保真度 {target_experimental_fidelity}")
            self.noise_model = calibrate_noise_for_fidelity(
                self.ideal_wigner, 
                target_fidelity=target_experimental_fidelity,
                base_noise_params=noise_params
            )
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
        
        # 历史记录
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
        
        # 最终结果
        self.final_pred = None
        self.final_variance = None
        self.final_F2 = None
        self.final_F3 = None
        self.final_ratio = None
    
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
            
            # 预测
            sparse_input = create_sparse_input(
                self.target_wigner, self.sampling_mask, self.noise_model
            )
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 计算保真度
            F2 = compute_fidelity(mean_pred, self.target_wigner)
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
    
    def plot_results(self, save_dir="results"):
        """绘制结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 实验态
        im0 = axes[0, 0].contourf(self.X, self.P, self.target_wigner, levels=40, cmap='RdBu_r')
        axes[0, 0].set_title(f"Experimental Target (F₁={self.F1_exp_vs_ideal:.4f})")
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 理想态
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
        
        # 6. 不确定性
        im5 = axes[1, 2].contourf(self.X, self.P, self.final_variance, levels=40, cmap='viridis')
        axes[1, 2].set_title("Uncertainty Map")
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.suptitle("Experimental GKP Tomography (F2 Optimization)", fontsize=14, fontweight='bold')
        
        filepath = f"{save_dir}/experimental_tomography_results.png"
        plt.savefig(filepath, dpi=300)
        print(f"\n保存结果图到 {filepath}")
        plt.close()
    
    def plot_committee_comparison(self, save_dir="results"):
        """绘制委员会成员对比"""
        os.makedirs(save_dir, exist_ok=True)
        
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
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for idx, (name, pred, fid) in enumerate(zip(
            [n for n, _ in self.models], individual_preds, individual_fidelities
        )):
            if idx < 5:
                row = idx // 3
                col = idx % 3
                im = axes[row, col].contourf(self.X, self.P, pred, levels=40, cmap='RdBu_r')
                axes[row, col].set_title(f"{name}\nF={fid:.5f}")
                axes[row, col].set_xlabel("x")
                axes[row, col].set_ylabel("p")
                plt.colorbar(im, ax=axes[row, col])
        
        # 委员会平均
        im = axes[1, 2].contourf(self.X, self.P, self.final_pred, levels=40, cmap='RdBu_r')
        axes[1, 2].set_title(f"Committee Average\nF={self.final_F2:.5f}")
        axes[1, 2].set_xlabel("x")
        axes[1, 2].set_ylabel("p")
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.suptitle("Individual Committee Member Performance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = f"{save_dir}/committee_comparison.png"
        plt.savefig(filepath, dpi=300)
        print(f"保存委员会对比图到 {filepath}")
        plt.close()
        
        # 打印统计
        print("\n" + "="*60)
        print("委员会成员单独表现:")
        print("="*60)
        for name, fid in zip([n for n, _ in self.models], individual_fidelities):
            print(f"  {name:20s}: F = {fid:.5f}")
        print(f"  {'Committee Average':20s}: F = {self.final_F2:.5f}")
        print(f"  {'Standard Deviation':20s}: σ = {np.std(individual_fidelities):.5f}")
        print(f"  {'Min Fidelity':20s}: {min(individual_fidelities):.5f}")
        print(f"  {'Max Fidelity':20s}: {max(individual_fidelities):.5f}")
        print("="*60)
