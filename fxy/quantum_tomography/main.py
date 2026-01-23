"""
量子态稀疏层析主入口

用法:
    python main.py [--state 1|2|3] [--alpha 2.0] [--n 1] [--parity even|odd]

参数:
    --state: 态类型 (1=Fock, 2=相干态, 3=猫态)
    --alpha: 相干态/猫态振幅
    --n: Fock态光子数
    --parity: 猫态奇偶性
"""

import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

from active_learning import QuantumTomography
from visualization import plot_all_results


def main():
    parser = argparse.ArgumentParser(description="量子态稀疏层析")
    parser.add_argument('--state', type=int, default=1, 
                       help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='相干态/猫态振幅')
    parser.add_argument('--n', type=int, default=3,
                       help='Fock态光子数')
    parser.add_argument('--parity', type=str, default='even',
                       help='猫态奇偶性: even/odd')
    parser.add_argument('--fidelity', type=float, default=0.95,
                       help='目标实验保真度')
    parser.add_argument('--rounds', type=int, default=50,
                       help='最大训练轮数')
    parser.add_argument('--epochs', type=int, default=80,
                       help='每轮训练epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='预训练epochs')
    parser.add_argument('--samples_per_round', type=int, default=10,
                       help='每轮增加的采样点数')
    args = parser.parse_args()
    
    # 设置随机种子
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
        state_params['parity'] = args.parity
    
    # 噪声参数
    noise_params = {
        'detection_efficiency': 0.90,
        'dark_count_rate': 0.005,
        'readout_noise_std': 0.015,
        'shot_noise_scale': 0.03,
        'calibration_drift': 0.005,
        'background_level': 0.003,
    }
    
    # 创建层析实例
    tomo = QuantumTomography(
        grid_size=64,
        state_type=args.state,
        initial_ratio=0.02,      # 初始2%
        samples_per_round=args.samples_per_round,  # 每轮增加点数
        max_rounds=args.rounds,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        lr=2e-3,
        target_fidelity=args.fidelity,
        F_threshold=0.99,
        noise_params=noise_params,
        committee_size=5,
        **state_params
    )
    
    # 运行层析
    tomo.run()
    
    # 生成6张可视化图
    plot_all_results(tomo, save_dir="results")
    
    print("\n" + "="*60)
    print("程序执行完成!")
    print("="*60)
    print(f"态类型: {tomo.state_name}")
    print(f"最终采样率: {tomo.final_ratio*100:.2f}%")
    print(f"F(重构 vs 实验): {tomo.final_F_exp:.5f}")
    print(f"F(重构 vs 理论): {tomo.final_F_ideal:.5f}")
    print(f"F(实验 vs 理论): {tomo.F_exp_vs_ideal:.5f}")
    print("="*60)


if __name__ == "__main__":
    main()
