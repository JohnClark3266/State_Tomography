"""
主动学习稀疏量子态层析模块

实现基于不确定性的主动采样策略：
- 初始采样2%
- 每轮增加2%
- 基于委员会预测方差选择新采样点
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from quantum_states import create_state
from neural_networks import build_model_pool, select_committee
from noise_model import ExperimentalNoise, calibrate_noise_for_fidelity, compute_fidelity


def create_sparse_input(wigner, mask, noise_model=None):
    """创建稀疏输入"""
    if noise_model is not None:
        noisy = noise_model.apply_measurement_noise(wigner, mask)
        sparse = np.where(mask, noisy, 0.0)
    else:
        sparse = np.where(mask, wigner, 0.0)
    
    mask_channel = mask.astype(np.float32)
    return np.stack([sparse, mask_channel], axis=0).astype(np.float32)


def generate_random_mask(grid_size, ratio):
    """生成随机采样掩码"""
    n = int(grid_size * grid_size * ratio)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    indices = np.random.choice(grid_size * grid_size, n, replace=False)
    mask.flat[indices] = True
    return mask


def generate_training_data(n_samples, grid_size, state_type, sampling_ratios,
                           noise_model=None, **state_params):
    """生成训练数据"""
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        # 生成随机参数变化的态
        params = state_params.copy()
        if state_type == 1:  # Fock
            params['n'] = np.random.randint(0, 5)
        elif state_type == 2:  # Coherent
            params['alpha'] = np.random.uniform(1.0, 3.0)
        elif state_type == 3:  # Cat
            params['alpha'] = np.random.uniform(1.5, 3.0)
            params['parity'] = np.random.choice(['even', 'odd'])
        
        _, _, ideal_wigner, _ = create_state(grid_size, state_type, **params)
        
        if noise_model:
            target_wigner = noise_model.apply_state_distortion(ideal_wigner)
        else:
            target_wigner = ideal_wigner.copy()
        
        ratio = np.random.uniform(sampling_ratios[0], sampling_ratios[1])
        mask = generate_random_mask(grid_size, ratio)
        sparse_input = create_sparse_input(target_wigner, mask, noise_model)
        
        inputs.append(sparse_input)
        targets.append(target_wigner[np.newaxis, :, :])
    
    return np.array(inputs), np.array(targets)


class QuantumTomography:
    """量子态层析主类"""
    
    def __init__(self, grid_size=64, state_type=1, 
                 initial_ratio=0.02, samples_per_round=20,
                 max_rounds=50, epochs=40, pretrain_epochs=100, lr=1e-3,
                 target_fidelity=0.95, F_threshold=0.99,
                 noise_params=None, committee_size=5,
                 **state_params):
        
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.initial_ratio = initial_ratio
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.F_threshold = F_threshold
        self.committee_size = committee_size
        
        # 设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 生成目标态
        self.X, self.P, self.ideal_wigner, self.state_name = create_state(
            grid_size, state_type, **state_params
        )
        print(f"目标态: {self.state_name}")
        
        # 噪声模型和实验态
        if noise_params is not None:
            print(f"\n校准噪声以产生目标保真度 {target_fidelity}")
            self.noise_model = calibrate_noise_for_fidelity(
                self.ideal_wigner, target_fidelity, noise_params
            )
            self.exp_wigner = self.noise_model.apply_state_distortion(self.ideal_wigner)
            self.F_exp_vs_ideal = compute_fidelity(self.exp_wigner, self.ideal_wigner)
            print(f"实验态创建完成: F(实验vs理论) = {self.F_exp_vs_ideal:.5f}")
        else:
            self.noise_model = None
            self.exp_wigner = self.ideal_wigner.copy()
            self.F_exp_vs_ideal = 1.0
        
        # 构建模型池并选择委员会
        print("\n构建神经网络模型池 (20个模型)...")
        self.model_pool = build_model_pool(grid_size)
        self.committee = select_committee(self.model_pool, committee_size)
        print(f"选中的委员会成员: {[n for n, _ in self.committee]}")
        
        for name, model in self.committee:
            model.to(self.device)
        
        # 采样掩码和历史
        self.sampling_mask = np.zeros((grid_size, grid_size), dtype=bool)
        self.sampling_history = []  # 每轮采样的掩码
        
        self.history = {
            'round': [],
            'sampling_ratio': [],
            'F_recon_vs_exp': [],
            'F_recon_vs_ideal': [],
            'mse': []
        }
        
        self.final_pred = None
        self.final_variance = None
        self.early_stopped = False
        self.stop_round = -1
        
    def _pretrain_committee(self):
        """
        使用指定态的理论形式进行预训练
        
        关键：使用self.ideal_wigner（用户指定的实验态对应的理论态）
        让委员会学习该特定量子态的结构，从而在初始阶段给出更合理的预测。
        """
        print(f"\n[预训练] 使用理论态预训练委员会 ({self.pretrain_epochs} epochs)...")
        print(f"  目标态: {self.state_name}")
        print("  策略: 稀疏重建学习 (Sparse -> Complete)")
            
        n_samples = 300
        inputs = []
        targets = []
        
        for _ in range(n_samples):
            # 随机采样率 2% - 30%
            ratio = np.random.uniform(0.02, 0.30)
            mask = generate_random_mask(self.grid_size, ratio)
            
            # 输入：理论态的稀疏采样 (可选加噪声增强鲁棒性)
            if self.noise_model and np.random.random() < 0.5:
                sparse_input = create_sparse_input(self.ideal_wigner, mask, self.noise_model)
            else:
                sparse_input = create_sparse_input(self.ideal_wigner, mask, None)
            
            inputs.append(sparse_input)
            # 目标始终是干净的理论态
            targets.append(self.ideal_wigner[np.newaxis, :, :])
            
        # 转换为Tensor
        train_inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        train_targets = torch.tensor(np.array(targets), dtype=torch.float32)
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            for _ in range(self.pretrain_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
        print("✓ 预训练完成\n")
    
    def _generate_training_data(self, n_samples=100):
        """生成训练数据 (随机掩码/Data Augmentation)"""
        current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        # 训练数据生成：使用带噪声的模拟实验数据逻辑
        # 使用随机掩码范围覆盖当前稀疏度，以增强模型对不同采样模式的鲁棒性
        return generate_training_data(
            n_samples, self.grid_size, self.state_type,
            (max(0.02, current_ratio - 0.05), min(0.35, current_ratio + 0.10)),
            self.noise_model, **self.state_params
        )
    
    def _train_committee(self, inputs, targets):
        train_inputs = torch.tensor(inputs, dtype=torch.float32)
        train_targets = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            # 对于在线学习，可以使用较小的学习率微调
            optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.5)
            for _ in range(self.epochs):
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
        for name, model in self.committee:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.var(axis=0), predictions
    
    def _compute_gradient_magnitude(self, pred):
        """计算预测的梯度幅值，用于识别变化剧烈区域"""
        # 使用Sobel算子计算梯度
        dy = np.abs(np.diff(pred, axis=0, prepend=pred[0:1, :]))
        dx = np.abs(np.diff(pred, axis=1, prepend=pred[:, 0:1]))
        gradient_mag = np.sqrt(dx**2 + dy**2)
        return gradient_mag
    
    def _compute_origin_weight(self):
        """计算距离原点的权重，原点附近物理特征通常更明显"""
        center = self.grid_size // 2
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        # 高斯权重，原点附近权重高
        sigma = self.grid_size / 4
        weight = np.exp(-dist**2 / (2 * sigma**2))
        return weight
    
    def _select_new_points(self, variance, n_points, mean_pred=None):
        """
        基于物理信息的采样点选择
        
        结合三个因素:
        1. 委员会方差 (不确定性) - 模型不确定的区域
        2. 梯度幅值 (变化剧烈区域) - Wigner函数振荡明显的地方
        3. 原点权重 (物理核心区域) - 量子态特征通常集中在相空间原点附近
        """
        # 归一化各因素到 [0, 1]
        var_score = variance.copy()
        var_score[self.sampling_mask] = 0  # 已采样点不考虑
        if var_score.max() > 0:
            var_score = var_score / var_score.max()
        
        # 梯度幅值
        if mean_pred is not None:
            grad_score = self._compute_gradient_magnitude(mean_pred)
            grad_score[self.sampling_mask] = 0
            if grad_score.max() > 0:
                grad_score = grad_score / grad_score.max()
        else:
            grad_score = np.zeros_like(variance)
        
        # 原点权重
        origin_weight = self._compute_origin_weight()
        origin_weight[self.sampling_mask] = 0
        
        # 综合评分: 不确定性 * 0.5 + 梯度 * 0.3 + 原点权重 * 0.2
        combined_score = 0.5 * var_score + 0.3 * grad_score + 0.2 * origin_weight
        combined_score[self.sampling_mask] = -np.inf  # 确保已采样点不被选中
        
        # 选择得分最高的n_points个点
        n_remaining = (~self.sampling_mask).sum()
        n_points = min(n_points, n_remaining)
        
        if n_points > 0:
            flat_indices = np.argsort(combined_score.ravel())[-n_points:]
            new_mask = np.zeros_like(self.sampling_mask)
            new_mask.flat[flat_indices] = True
            return new_mask
        else:
            return np.zeros_like(self.sampling_mask)
    
    def run(self):
        """运行主动学习层析"""
        print("\n" + "="*60)
        print(f"主动学习稀疏量子态层析 (智能初始化采样)")
        print("="*60)
        print(f"态类型: {self.state_name}")
        print(f"初始采样率: {self.initial_ratio*100:.1f}%")
        print(f"每轮增加点数: {self.samples_per_round}")
        print(f"预训练Epochs: {self.pretrain_epochs}")
        print(f"目标保真度: F ≥ {self.F_threshold}")
        print("="*60 + "\n")
        
        # 0. 预训练 (通用结构学习)
        self._pretrain_committee()
        
        # 1. 基于预训练结果的智能初始采样
        # 使用预训练后的委员会对空输入（或极稀疏输入）进行预测
        # 根据委员会不确定性（方差）选择初始采样点
        print("[初始化] 基于预训练结果选择初始采样点...")
        
        # 创建一个极稀疏的输入（仅几个随机点）作为"探测"
        probe_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        # 在四个角和中心放几个探测点
        corner_offset = 5
        probe_mask[corner_offset, corner_offset] = True
        probe_mask[corner_offset, -corner_offset-1] = True
        probe_mask[-corner_offset-1, corner_offset] = True
        probe_mask[-corner_offset-1, -corner_offset-1] = True
        probe_mask[self.grid_size//2, self.grid_size//2] = True
        
        # 使用理论态的这些点（模拟真实实验时会用实验测量值）
        probe_input = create_sparse_input(self.ideal_wigner, probe_mask, None)
        
        # 让预训练后的委员会预测并计算不确定性
        mean_pred, variance, _ = self._committee_predict(probe_input)
        
        # 计算综合得分（不确定性 + 梯度 + 原点权重）
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        
        # 根据预测的方差和梯度选择初始点
        initial_mask = self._select_new_points(variance, n_initial, mean_pred=mean_pred)
        
        # 确保探测点也被包含在初始采样中
        initial_mask |= probe_mask
        
        print(f"  根据预训练不确定性选择了 {initial_mask.sum()} 个初始采样点")
        print(f"  采样偏向: 变化剧烈区域 + 原点附近\n")
            
        self.sampling_mask = initial_mask.copy()
        self.sampling_history.append(initial_mask.copy())
        
        for round_id in range(self.max_rounds):
            current_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
            n_current = self.sampling_mask.sum()
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样点: {n_current} ({current_ratio*100:.2f}%)")
            
            # 训练 (使用随机掩码进行增广训练，提高泛化性)
            # n_samples=200 保证足够的训练量
            inputs, targets = self._generate_training_data(n_samples=200)
            self._train_committee(inputs, targets)
            
            # 预测
            sparse_input = create_sparse_input(
                self.exp_wigner, self.sampling_mask, self.noise_model
            )
            mean_pred, variance, _ = self._committee_predict(sparse_input)
            
            # 强制使用采样点的测量值 (Hard Constraint)
            measured_values = sparse_input[0]
            mean_pred[self.sampling_mask] = measured_values[self.sampling_mask]
            
            # 计算保真度
            F_recon_exp = compute_fidelity(mean_pred, self.exp_wigner)
            F_recon_ideal = compute_fidelity(mean_pred, self.ideal_wigner)
            mse = np.mean((mean_pred - self.exp_wigner) ** 2)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(current_ratio)
            self.history['F_recon_vs_exp'].append(F_recon_exp)
            self.history['F_recon_vs_ideal'].append(F_recon_ideal)
            self.history['mse'].append(mse)
            
            print(f"  F(重构vs实验): {F_recon_exp:.5f}")
            print(f"  F(重构vs理论): {F_recon_ideal:.5f}")
            print(f"  MSE: {mse:.6f}")
            
            # 早停
            if F_recon_exp >= self.F_threshold:
                print(f"\n✓ 保真度达到阈值 {self.F_threshold}，提前停止！")
                self.early_stopped = True
                self.stop_round = round_id + 1
                break
            
            # 添加新采样点 (基于物理信息的智能采样)
            if round_id < self.max_rounds - 1:
                n_new = self.samples_per_round
                # 传入mean_pred以计算梯度权重
                new_mask = self._select_new_points(variance, n_new, mean_pred=mean_pred)
                self.sampling_mask |= new_mask
                self.sampling_history.append(new_mask.copy())
        
        # 最终预测
        sparse_input = create_sparse_input(
            self.exp_wigner, self.sampling_mask, self.noise_model
        )
        self.final_pred, self.final_variance, _ = self._committee_predict(sparse_input)
        
        # 强制应用最终测量值
        measured_values = sparse_input[0]
        self.final_pred[self.sampling_mask] = measured_values[self.sampling_mask]
        
        self.final_F_exp = compute_fidelity(self.final_pred, self.exp_wigner)
        self.final_F_ideal = compute_fidelity(self.final_pred, self.ideal_wigner)
        self.final_ratio = self.sampling_mask.sum() / (self.grid_size ** 2)
        
        print("\n" + "="*60)
        print("实验结束总结:")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        print(f"F(重构vs实验): {self.final_F_exp:.5f}")
        print(f"F(重构vs理论): {self.final_F_ideal:.5f}")
        print(f"F(实验vs理论): {self.F_exp_vs_ideal:.5f}")
        if self.early_stopped:
            print(f"提前停止于第 {self.stop_round} 轮")
        print("="*60)
