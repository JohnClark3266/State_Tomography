"""
快速主动学习层析模块

使用新的模块化架构:
- SamplingManager: 管理采样状态矩阵
- DecisionMaker: 决定下一轮采样点
- 优化训练: 预训练后只做轻量微调

流程:
1. 预训练委员会
2. 用委员会预测初始不确定性，决定初始采样点
3. 每轮:
   a. 执行采样 (SamplingManager)
   b. 委员会预测
   c. 决定下一轮采样点 (DecisionMaker)
4. 最终填充预测值
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sampling_manager import SamplingManager
from decision_maker import DecisionMaker
from quantum_states import create_state
from neural_networks import build_model_pool, select_committee
from noise_model import compute_fidelity


class FastTomography:
    """快速主动学习层析"""
    
    def __init__(self, grid_size=64, state_type=2,
                 initial_ratio=0.02, samples_per_round=10,
                 max_rounds=50, pretrain_epochs=100, finetune_epochs=20,
                 lr=1e-3, F_threshold=0.99, committee_size=5,
                 noise_std=0.02, **state_params):
        
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.initial_ratio = initial_ratio
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.F_threshold = F_threshold
        self.noise_std = noise_std
        
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
        
        # 模拟实验态 (加噪声)
        self.exp_wigner = self.ideal_wigner + np.random.normal(0, noise_std, self.ideal_wigner.shape)
        
        # 初始化管理器
        self.sampler = SamplingManager(grid_size)
        self.decision = DecisionMaker(grid_size)
        
        # 构建委员会
        print("\n构建神经网络模型池...")
        self.model_pool = build_model_pool(grid_size)
        self.committee = select_committee(self.model_pool, committee_size)
        print(f"选中的委员会成员: {[n for n, _ in self.committee]}")
        
        for name, model in self.committee:
            model.to(self.device)
        
        # 历史记录
        self.history = {
            'round': [],
            'sampling_ratio': [],
            'F_recon_vs_exp': [],
            'F_recon_vs_ideal': [],
        }
        
        self.final_pred = None
    
    def _pretrain(self):
        """预训练: 学习从稀疏到完整的重建"""
        print(f"\n[预训练] {self.pretrain_epochs} epochs...")
        
        # 生成预训练数据
        n_samples = 200
        inputs = []
        targets = []
        
        for _ in range(n_samples):
            ratio = np.random.uniform(0.02, 0.30)
            n_sample = int(self.grid_size ** 2 * ratio)
            indices = np.random.choice(self.grid_size ** 2, n_sample, replace=False)
            
            # 稀疏输入
            sparse = np.zeros((self.grid_size, self.grid_size))
            mask = np.zeros((self.grid_size, self.grid_size))
            
            flat_sparse = sparse.flatten()
            flat_mask = mask.flatten()
            flat_sparse[indices] = self.ideal_wigner.flatten()[indices]
            flat_mask[indices] = 1.0
            
            sparse = flat_sparse.reshape(self.grid_size, self.grid_size)
            mask = flat_mask.reshape(self.grid_size, self.grid_size)
            
            inp = np.stack([sparse, mask], axis=0).astype(np.float32)
            inputs.append(inp)
            targets.append(self.ideal_wigner[np.newaxis, :, :].astype(np.float32))
        
        # 训练
        train_inputs = torch.tensor(np.array(inputs))
        train_targets = torch.tensor(np.array(targets))
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
    
    def _committee_predict(self, sparse_input):
        """委员会预测"""
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :], dtype=torch.float32).to(self.device)
        
        predictions = []
        for name, model in self.committee:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.var(axis=0)
    
    def _light_finetune(self):
        """轻量微调 (使用当前稀疏数据)"""
        sparse_input = self.sampler.get_sparse_input_for_nn()
        target = self.ideal_wigner[np.newaxis, :, :].astype(np.float32)
        
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :]).to(self.device)
        target_tensor = torch.tensor(target[np.newaxis, :, :, :]).to(self.device)
        
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.1)
            for _ in range(self.finetune_epochs):
                optimizer.zero_grad()
                pred = model(input_tensor)
                loss = criterion(pred, target_tensor)
                loss.backward()
                optimizer.step()
    
    def run(self):
        """运行快速层析"""
        print("\n" + "="*60)
        print("快速主动学习层析 (模块化架构)")
        print("="*60)
        print(f"态类型: {self.state_name}")
        print(f"每轮增加点数: {self.samples_per_round}")
        print("="*60 + "\n")
        
        # 1. 预训练
        self._pretrain()
        
        # 2. 初始采样决策
        print("[初始化] 基于预训练结果选择初始采样点...")
        
        # 用空输入让委员会预测，获取不确定性
        empty_input = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        mean_pred, variance = self._committee_predict(empty_input)
        
        # 决定初始采样点
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = self.decision.decide_initial_samples(
            variance.flatten(), mean_pred.flatten(), n_initial
        )
        
        # 设置状态为1
        self.sampler.set_points_to_sample(initial_indices)
        
        # 执行初始采样
        self.sampler.execute_sampling(self.exp_wigner, noise_model=None)
        print(f"  初始采样: {self.sampler.get_sampled_count()} 点\n")
        
        # 3. 主循环
        for round_id in range(self.max_rounds):
            ratio = self.sampler.get_sampling_ratio()
            n_sampled = self.sampler.get_sampled_count()
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样点: {n_sampled} ({ratio*100:.2f}%)")
            
            # 每轮微调以持续提高保真度
            self._light_finetune()
            
            # 委员会预测
            sparse_input = self.sampler.get_sparse_input_for_nn()
            mean_pred, variance = self._committee_predict(sparse_input)
            
            # 强制使用已采样点的测量值
            mask = self.sampler.get_mask_2d()
            measured = self.sampler.get_wigner_2d()
            mean_pred[mask] = measured[mask]
            
            # 计算保真度
            F_exp = compute_fidelity(mean_pred, self.exp_wigner)
            F_ideal = compute_fidelity(mean_pred, self.ideal_wigner)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(ratio)
            self.history['F_recon_vs_exp'].append(F_exp)
            self.history['F_recon_vs_ideal'].append(F_ideal)
            
            print(f"  F(vs实验): {F_exp:.5f}  F(vs理论): {F_ideal:.5f}")
            
            # 早停
            if F_exp >= self.F_threshold:
                print(f"\n✓ 达到目标保真度 {self.F_threshold}!")
                break
            
            # 决定下一轮采样点
            if round_id < self.max_rounds - 1:
                current_state = self.sampler.get_state()
                next_indices = self.decision.decide_next_samples(
                    variance.flatten(), mean_pred.flatten(),
                    current_state, self.samples_per_round
                )
                
                # 设置状态为1，然后执行采样
                self.sampler.set_points_to_sample(next_indices)
                self.sampler.execute_sampling(self.exp_wigner, noise_model=None)
        
        # 4. 最终预测
        sparse_input = self.sampler.get_sparse_input_for_nn()
        self.final_pred, _ = self._committee_predict(sparse_input)
        
        # 填充预测值到未采样点
        self.sampler.fill_predictions(self.final_pred)
        
        # 已采样点用测量值
        mask = self.sampler.get_mask_2d()
        measured = self.sampler.get_wigner_2d()
        self.final_pred[mask] = measured[mask]
        
        # 设置兼容visualization.py的属性
        self.sampling_mask = mask
        self.final_ratio = self.sampler.get_sampling_ratio()
        self.final_F_exp = compute_fidelity(self.final_pred, self.exp_wigner)
        self.final_F_ideal = compute_fidelity(self.final_pred, self.ideal_wigner)
        self.F_exp_vs_ideal = compute_fidelity(self.exp_wigner, self.ideal_wigner)
        
        # 将sampler的采样历史转换为mask格式
        self.sampling_history = []
        cumulative_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for indices in self.sampler.sampling_history:
            round_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for idx in indices:
                i, j = idx // self.grid_size, idx % self.grid_size
                round_mask[i, j] = True
            cumulative_mask |= round_mask
            self.sampling_history.append(cumulative_mask.copy())
        
        print("\n" + "="*60)
        print("完成!")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        print(f"F(vs实验): {self.final_F_exp:.5f}")
        print(f"F(vs理论): {self.final_F_ideal:.5f}")
        print("="*60)
        
        return self.final_pred
