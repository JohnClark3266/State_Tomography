"""
快速层析主入口

用法:
    python main_fast.py --state 2 --alpha 2.0 --samples 10
"""

import argparse
import numpy as np
import torch

from fast_tomography import FastTomography
from visualization import plot_all_results


def main():
    parser = argparse.ArgumentParser(description="快速量子态层析")
    parser.add_argument('--state', type=int, default=2, help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=float, default=2.0, help='相干态/猫态振幅')
    parser.add_argument('--n', type=int, default=3, help='Fock态光子数')
    parser.add_argument('--samples', type=int, default=10, help='每轮增加的采样点数')
    parser.add_argument('--rounds', type=int, default=50, help='最大轮数')
    parser.add_argument('--pretrain', type=int, default=30, help='预训练epochs')
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
    tomo = FastTomography(
        grid_size=64,
        state_type=args.state,
        initial_ratio=0.02,
        samples_per_round=args.samples,
        max_rounds=args.rounds,
        pretrain_epochs=args.pretrain,
        finetune_epochs=30,
        lr=2e-3,
        F_threshold=0.98,
        committee_size=5,
        noise_std=0.006,
        **state_params
    )
    
    # 运行
    tomo.run()
    
    # 可视化 (调用visualization.py)
    plot_all_results(tomo, save_dir="results")


if __name__ == "__main__":
    main()
