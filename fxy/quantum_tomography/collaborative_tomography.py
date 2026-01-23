"""
协作模式层析

与MATLAB用户协作的完整工作流:
1. 预训练模型
2. 决定初始采样点
3. 导出采样请求给MATLAB
4. [等待合作者测量]
5. 导入实验数据
6. 训练模型并决定下一轮采样点
7. 重复3-6直到达到目标保真度

用法:
    # 第一次运行 (导出初始采样请求)
    python collaborative_tomography.py --mode init --state 2 --alpha 2.0
    
    # 导入数据并继续 (每轮)
    python collaborative_tomography.py --mode continue --import_file matlab_exchange/exp_data.mat
"""

import argparse
import numpy as np
import torch
import os

from fast_tomography import FastTomography
from matlab_interface import MatlabInterface, export_for_matlab, import_from_matlab
from visualization import plot_all_results
from noise_model import compute_fidelity


class CollaborativeTomography:
    """协作模式层析"""
    
    def __init__(self, grid_size=64, state_type=2, 
                 samples_per_round=10, F_threshold=0.98,
                 save_dir="matlab_exchange", **state_params):
        
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.samples_per_round = samples_per_round
        self.F_threshold = F_threshold
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化FastTomography (用于模型和采样管理)
        self.tomo = FastTomography(
            grid_size=grid_size,
            state_type=state_type,
            samples_per_round=samples_per_round,
            max_rounds=1,  # 协作模式下每次只做一轮
            pretrain_epochs=30,
            finetune_epochs=30,
            lr=2e-3,
            F_threshold=F_threshold,
            noise_std=0.0,  # 实验数据已有噪声
            **state_params
        )
        
        # MATLAB接口
        self.matlab = MatlabInterface(grid_size, save_dir)
        
        # 轮次计数
        self.round_id = 0
        self.state_file = os.path.join(save_dir, "collaboration_state.npz")
    
    def initialize(self):
        """
        初始化: 预训练并决定初始采样点
        """
        print("\n" + "="*60)
        print("协作模式初始化")
        print("="*60)
        
        # 预训练
        self.tomo._pretrain()
        
        # 决定初始采样点
        print("[初始化] 确定初始采样点...")
        empty_input = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        mean_pred, variance = self.tomo._committee_predict(empty_input)
        
        n_initial = int(self.grid_size ** 2 * 0.02)  # 2%初始点
        initial_indices = self.tomo.decision.decide_initial_samples(
            variance.flatten(), mean_pred.flatten(), n_initial
        )
        
        # 设置状态为1 (待采样)
        self.tomo.sampler.set_points_to_sample(initial_indices)
        
        self.round_id = 1
        self._save_state()
        
        # 导出给MATLAB
        filepath = self.matlab.export_sampling_request(
            self.tomo.sampler, round_id=self.round_id
        )
        
        return filepath
    
    def process_experimental_data(self, import_filepath: str):
        """
        处理实验数据: 导入 -> 训练 -> 决定下一轮 -> 导出
        
        参数:
            import_filepath: MATLAB返回的数据文件路径
        """
        print("\n" + "="*60)
        print(f"处理实验数据 (Round {self.round_id})")
        print("="*60)
        
        # 导入实验数据
        n_new, n_total = import_from_matlab(import_filepath, self.tomo.sampler)
        
        # 微调训练
        print("\n[训练] 微调模型...")
        self.tomo._light_finetune()
        
        # 委员会预测
        sparse_input = self.tomo.sampler.get_sparse_input_for_nn()
        mean_pred, variance = self.tomo._committee_predict(sparse_input)
        
        # 强制使用测量值
        mask = self.tomo.sampler.get_mask_2d()
        measured = self.tomo.sampler.get_wigner_2d()
        mean_pred[mask] = measured[mask]
        
        # 评估当前保真度 (如果有理论态)
        F_ideal = compute_fidelity(mean_pred, self.tomo.ideal_wigner)
        sampling_ratio = n_total / (self.grid_size ** 2)
        
        print(f"\n[结果] Round {self.round_id}")
        print(f"  采样点: {n_total} ({sampling_ratio*100:.2f}%)")
        print(f"  F(vs理论): {F_ideal:.5f}")
        
        # 检查是否达到目标
        if F_ideal >= self.F_threshold:
            print(f"\n✓ 达到目标保真度 {self.F_threshold}!")
            self.tomo.final_pred = mean_pred
            self._finalize()
            return None
        
        # 决定下一轮采样点
        self.round_id += 1
        current_state = self.tomo.sampler.get_state()
        next_indices = self.tomo.decision.decide_next_samples(
            variance.flatten(), mean_pred.flatten(),
            current_state, self.samples_per_round
        )
        
        # 设置状态为1
        self.tomo.sampler.set_points_to_sample(next_indices)
        
        self._save_state()
        
        # 导出下一轮采样请求
        filepath = self.matlab.export_sampling_request(
            self.tomo.sampler, round_id=self.round_id
        )
        
        return filepath
    
    def _save_state(self):
        """保存当前状态"""
        np.savez(self.state_file,
                 round_id=self.round_id,
                 state_type=self.state_type,
                 grid_size=self.grid_size)
        print(f"状态已保存: {self.state_file}")
    
    def _load_state(self):
        """加载状态"""
        if os.path.exists(self.state_file):
            data = np.load(self.state_file)
            self.round_id = int(data['round_id'])
            print(f"状态已加载: Round {self.round_id}")
    
    def _finalize(self):
        """完成协作，生成最终结果"""
        print("\n" + "="*60)
        print("协作完成!")
        print("="*60)
        
        # 设置兼容visualization.py的属性
        mask = self.tomo.sampler.get_mask_2d()
        self.tomo.sampling_mask = mask
        self.tomo.final_ratio = self.tomo.sampler.get_sampling_ratio()
        self.tomo.final_F_exp = compute_fidelity(self.tomo.final_pred, self.tomo.exp_wigner)
        self.tomo.final_F_ideal = compute_fidelity(self.tomo.final_pred, self.tomo.ideal_wigner)
        self.tomo.F_exp_vs_ideal = compute_fidelity(self.tomo.exp_wigner, self.tomo.ideal_wigner)
        self.tomo.sampling_history = [mask]
        
        # 可视化
        plot_all_results(self.tomo, save_dir="results")


def main():
    parser = argparse.ArgumentParser(description="协作模式层析")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['init', 'continue'],
                        help='init=初始化, continue=继续')
    parser.add_argument('--import_file', type=str, default=None,
                        help='MATLAB返回的数据文件路径')
    parser.add_argument('--state', type=int, default=2, 
                        help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='相干态/猫态振幅')
    parser.add_argument('--n', type=int, default=3, 
                        help='Fock态光子数')
    parser.add_argument('--samples', type=int, default=10,
                        help='每轮采样点数')
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
    
    # 创建协作层析实例
    collab = CollaborativeTomography(
        grid_size=64,
        state_type=args.state,
        samples_per_round=args.samples,
        F_threshold=0.98,
        **state_params
    )
    
    if args.mode == 'init':
        # 初始化模式
        filepath = collab.initialize()
        print(f"\n下一步: 将 {filepath} 发送给MATLAB用户")
        
    elif args.mode == 'continue':
        # 继续模式
        if args.import_file is None:
            print("错误: continue模式需要 --import_file 参数")
            return
        
        collab._load_state()
        next_file = collab.process_experimental_data(args.import_file)
        
        if next_file:
            print(f"\n下一步: 将 {next_file} 发送给MATLAB用户")
        else:
            print("\n协作完成! 结果已保存到 results/")


if __name__ == "__main__":
    main()
