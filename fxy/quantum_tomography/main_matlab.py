"""
MATLAB 集成层析主入口

使用 MATLAB Engine 进行真实 Wigner 函数采样，完整运行主动学习层析流程。

用法:
    # 确保 MATLAB 已运行并执行了 matlab.engine.shareEngine
    python main_matlab.py --state 2 --alpha 2.0 --samples 20

与 main_fast.py 的区别:
    - sampling_manager.execute_sampling() -> execute_sampling_matlab()
    - 所有 Wigner 值由 MATLAB 计算返回
"""

import argparse
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
from visualization import plot_all_results
from matlab_bridge import MatlabBridge


class FastTomographyMatlab:
    """使用 MATLAB 采样的快速主动学习层析"""
    
    def __init__(self, grid_size=64, state_type=2,
                 initial_ratio=0.0073, samples_per_round=20,  # 0.0073 ≈ 30 初始点
                 max_rounds=50, pretrain_epochs=30, finetune_epochs=30,
                 lr=2e-3, F_threshold=0.98, committee_size=5,
                 noise_std=0.006, **state_params):
        
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
        
        # 连接 MATLAB
        print("\n连接 MATLAB...")
        self.matlab_bridge = MatlabBridge()
        if not self.matlab_bridge.is_connected:
            raise RuntimeError("无法连接 MATLAB! 请确保 MATLAB 已运行并执行了 matlab.engine.shareEngine")
        
        # 生成目标态 (Python 理论值，用于预训练和初始化)
        self.X, self.P, self.ideal_wigner, self.state_name = create_state(
            grid_size, state_type, **state_params
        )
        print(f"目标态: {self.state_name}")
        
        # 使用 Python 理论态作为参考 (初始化不调用 MATLAB)
        # 实际采样时 MATLAB 会计算并返回真实值
        self.exp_wigner = self.ideal_wigner.copy()
        
        # 初始化管理器 (with MATLAB bridge)
        self.sampler = SamplingManager(grid_size, matlab_bridge=self.matlab_bridge)
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
    
    def _get_matlab_full_wigner(self):
        """从 MATLAB 获取完整的 Wigner 函数（无噪声）作为真值"""
        # 发送全部网格点坐标
        self.matlab_bridge.send('full_X', self.X)
        self.matlab_bridge.send('full_P', self.P)
        
        alpha = self.state_params.get('alpha', 2.0)
        n = self.state_params.get('n', 3)
        
        if self.state_type == 2:  # 相干态
            matlab_code = f'''
                alpha = {alpha};
                sqrt2 = sqrt(2);
                x0 = 0;
                p0 = sqrt2 * alpha;
                matlab_wigner = (1/pi) * exp(-((full_X - x0).^2 + (full_P - p0).^2));
            '''
        elif self.state_type == 1:  # Fock 态
            matlab_code = f'''
                n = {n};
                r2 = full_X.^2 + full_P.^2;
                L_n = laguerreL(n, 0, 2*r2);
                matlab_wigner = ((-1)^n / pi) * L_n .* exp(-r2);
            '''
        elif self.state_type == 3:  # 猫态
            matlab_code = f'''
                alpha = {alpha};
                sqrt2 = sqrt(2);
                W_plus = (1/pi) * exp(-((full_X - sqrt2*alpha).^2 + full_P.^2));
                W_minus = (1/pi) * exp(-((full_X + sqrt2*alpha).^2 + full_P.^2));
                interference = (2/pi) * exp(-(full_X.^2 + full_P.^2)) .* cos(2*sqrt2*alpha*full_P);
                N2_even = 2 * (1 + exp(-2*alpha^2));
                matlab_wigner = (W_plus + W_minus + interference * exp(-alpha^2)) / N2_even;
            '''
        else:
            return self.ideal_wigner.copy()
        
        self.matlab_bridge.eval(matlab_code)
        matlab_wigner = self.matlab_bridge.receive('matlab_wigner')
        
        if matlab_wigner is not None:
            return matlab_wigner.astype(np.float32)
        else:
            print("  ⚠ 无法从MATLAB获取Wigner函数，使用Python版本")
            return self.ideal_wigner.copy()
    
    def _pretrain(self):
        """预训练: 直接用完整理论态训练一次"""
        print("\n[预训练] 使用完整理论态 (4096 点)...")
        
        # 输入: 完整 Wigner + 全1掩码
        full_wigner = self.ideal_wigner.astype(np.float32)
        full_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        # (1, 2, 64, 64): batch=1, channels=2
        train_input = torch.tensor(
            np.stack([full_wigner, full_mask], axis=0)[np.newaxis, :, :, :]
        ).to(self.device)
        
        # 目标: 完整 Wigner (1, 1, 64, 64)
        train_target = torch.tensor(
            full_wigner[np.newaxis, np.newaxis, :, :]
        ).to(self.device)
        
        criterion = nn.MSELoss()
        
        # 对每个模型进行单次训练
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            optimizer.zero_grad()
            pred = model(train_input)
            loss = criterion(pred, train_target)
            loss.backward()
            optimizer.step()
            
            print(f"  {name}: loss = {loss.item():.6f}")
        
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
        """运行快速层析 (MATLAB 采样版)"""
        print("\n" + "="*60)
        print("快速主动学习层析 (MATLAB 采样)")
        print("="*60)
        print(f"态类型: {self.state_name}")
        print(f"每轮增加点数: {self.samples_per_round}")
        print(f"目标保真度: {self.F_threshold}")
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
        
        # 使用 MATLAB 执行初始采样
        print(f"  [MATLAB] 采样 {len(initial_indices)} 个初始点...")
        success = self.sampler.execute_sampling_matlab(
            state_type=self.state_type,
            noise_std=self.noise_std,
            **self.state_params
        )
        if not success:
            print("  ⚠ MATLAB 采样失败，使用 Python 备份")
            self.sampler.execute_sampling(self.exp_wigner, noise_model=None)
        
        print(f"  ✓ 初始采样: {self.sampler.get_sampled_count()} 点\n")
        
        # 3. 主循环
        for round_id in range(self.max_rounds):
            ratio = self.sampler.get_sampling_ratio()
            n_sampled = self.sampler.get_sampled_count()
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样点: {n_sampled} ({ratio*100:.2f}%)")
            
            # 每轮微调
            self._light_finetune()
            
            # 委员会预测
            sparse_input = self.sampler.get_sparse_input_for_nn()
            mean_pred, variance = self._committee_predict(sparse_input)
            
            # 强制使用已采样点的测量值
            mask = self.sampler.get_mask_2d()
            measured = self.sampler.get_wigner_2d()
            mean_pred[mask] = measured[mask]
            
            # 计算保真度
            # F_exp: 对比 MATLAB 计算的完整 Wigner 函数
            # F_ideal: 对比 Python 计算的理论 Wigner 函数
            F_exp = compute_fidelity(mean_pred, self.exp_wigner)  # vs MATLAB
            F_ideal = compute_fidelity(mean_pred, self.ideal_wigner)  # vs Python
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(ratio)
            self.history['F_recon_vs_exp'].append(F_exp)
            self.history['F_recon_vs_ideal'].append(F_ideal)
            
            print(f"  F(vs MATLAB): {F_exp:.5f}  F(vs Python): {F_ideal:.5f}")
            
            # 早停 (使用 MATLAB 保真度作为主要指标)
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
                
                # 设置状态为1，然后用 MATLAB 采样
                self.sampler.set_points_to_sample(next_indices)
                
                print(f"  [MATLAB] 采样 {len(next_indices)} 个新点...")
                success = self.sampler.execute_sampling_matlab(
                    state_type=self.state_type,
                    noise_std=self.noise_std,
                    **self.state_params
                )
                if not success:
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
        
        # 设置兼容 visualization.py 的属性
        self.sampling_mask = mask
        self.final_ratio = self.sampler.get_sampling_ratio()
        self.final_F_exp = compute_fidelity(self.final_pred, self.exp_wigner)
        self.final_F_ideal = compute_fidelity(self.final_pred, self.ideal_wigner)
        self.F_exp_vs_ideal = compute_fidelity(self.exp_wigner, self.ideal_wigner)
        
        # 将 sampler 的采样历史转换为 mask 格式
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
        
        # 关闭 MATLAB 连接
        self.matlab_bridge.close()
        
        return self.final_pred


def main():
    parser = argparse.ArgumentParser(description="MATLAB 集成量子态层析")
    parser.add_argument('--state', type=int, default=2, help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=float, default=2.0, help='相干态/猫态振幅')
    parser.add_argument('--n', type=int, default=3, help='Fock态光子数')
    parser.add_argument('--samples', type=int, default=20, help='每轮增加的采样点数')
    parser.add_argument('--rounds', type=int, default=50, help='最大轮数')
    parser.add_argument('--pretrain', type=int, default=30, help='预训练epochs')
    parser.add_argument('--finetune', type=int, default=30, help='微调epochs')
    parser.add_argument('--threshold', type=float, default=0.98, help='目标保真度')
    args = parser.parse_args()
    
    # 设置种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 态参数
    state_params = {}
    if args.state == 1:
        state_params['n'] = args.n
    elif args.state == 2:
        state_params['alpha'] = args.alpha
    elif args.state == 3:
        state_params['alpha'] = args.alpha
        state_params['parity'] = 'even'
    
    # 创建层析实例
    tomo = FastTomographyMatlab(
        grid_size=64,
        state_type=args.state,
        initial_ratio=0.0073,  # ~30 个初始采样点
        samples_per_round=args.samples,
        max_rounds=args.rounds,
        pretrain_epochs=30,  # 固定为30
        finetune_epochs=args.finetune,
        lr=2e-3,
        F_threshold=args.threshold,
        committee_size=5,
        noise_std=0.006,
        **state_params
    )
    
    # 运行
    tomo.run()
    
    # 可视化
    plot_all_results(tomo, save_dir="results")
    print("\n结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()
