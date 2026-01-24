"""
GKP态稀疏量子态层析 - 主入口

基于主动学习的稀疏量子态层析实验模拟。
使用CNN委员会进行状态重建，支持实验噪声模拟。

用法:
    python main.py
"""

import sys
import os
import warnings
import numpy as np
import torch

# 添加父目录到路径，支持直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gkp_state import create_gkp_grid
from noise_model import ExperimentalNoise, calibrate_noise_for_fidelity
from sparse_sampling import create_sparse_input, generate_training_data
from cnn_models import build_cnn_committee
from fidelity import compute_fidelity
from active_learning import ActiveSparseTomography


warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def main():
    """主函数 - 实验数据模拟"""
    
    # 实验噪声参数（可调整以模拟不同实验条件）
    noise_params = {
        'detection_efficiency': 0.85,      # 85% 探测效率
        'dark_count_rate': 0.01,           # 1% 暗计数
        'readout_noise_std': 0.02,         # 2% 读出噪声
        'shot_noise_scale': 0.05,          # 5% 散粒噪声
        'calibration_drift': 0.01,         # 1% 校准漂移
        'background_level': 0.005,         # 0.5% 背景噪声
    }
    
    tomography = ActiveSparseTomography(
        grid_size=64,                # 64x64分辨率
        initial_ratio=0.03,          # 初始3%采样
        add_ratio=0.015,             # 每轮增加1.5%
        max_rounds=30,               # 最多30轮
        epochs=30,                   # 50个epoch
        lr=2e-3,                     # 学习率
        target_delta=0.3,            # GKP参数
        noise_params=noise_params,   # 实验噪声
        target_experimental_fidelity=0.95,  # 目标F1
        F2_threshold=0.99            # 目标F2
    )
    
    tomography.run()
    tomography.plot_results()
    tomography.plot_committee_comparison()
    
    print("\n程序执行完成!")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("实验统计:")
    print(f"  总采样点数: {tomography.sampling_mask.sum()}")
    print(f"  采样率: {tomography.final_ratio*100:.2f}%")
    print(f"  F1 (实验 vs 理论): {tomography.F1_exp_vs_ideal:.5f}")
    print(f"  F2 (重建 vs 实验): {tomography.final_F2:.5f}")
    print(f"  F3 (重建 vs 理论): {tomography.final_F3:.5f}")
    print(f"  训练轮数: {len(tomography.history['round'])}")
    if tomography.early_stopped:
        print(f"  早停: 是 (第{tomography.stop_round}轮)")
    else:
        print(f"  早停: 否")
    print("="*60)


if __name__ == "__main__":
    main()
