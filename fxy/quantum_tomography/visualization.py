"""
可视化模块

生成6张可视化图：
1. 采样分布（每轮不同颜色，理论态衬底）
2. 保真度曲线
3. 重构Wigner函数
4. 实验态Wigner函数
5. 相对误差百分比
6. 采样密度热力图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_all_results(tomo, save_dir="results"):
    """
    生成全部6张可视化图
    
    参数:
        tomo: QuantumTomography实例
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 图1: 采样分布
    plot_sampling_distribution(tomo, save_dir)
    
    # 图2: 保真度曲线
    plot_fidelity_curves(tomo, save_dir)
    
    # 图3: 重构Wigner函数
    plot_reconstruction(tomo, save_dir)
    
    # 图4: 实验态Wigner函数
    plot_experimental_state(tomo, save_dir)
    
    # 图5: 相对误差
    plot_relative_error(tomo, save_dir)
    
    # 图6: 采样密度热力图
    plot_sampling_density(tomo, save_dir)
    
    print(f"\n所有结果图已保存到 {save_dir}/")


def plot_sampling_distribution(tomo, save_dir):
    """图1: 采样分布（每轮不同颜色）"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 半透明理论态衬底
    vmax = np.max(np.abs(tomo.ideal_wigner))
    ax.contourf(tomo.X, tomo.P, tomo.ideal_wigner, levels=40, 
                cmap='RdBu_r', alpha=0.3, vmin=-vmax, vmax=vmax)
    
    # 为每轮采样点分配不同颜色
    n_rounds = len(tomo.sampling_history)
    colors = plt.cm.viridis(np.linspace(0, 1, n_rounds))
    
    for i, mask in enumerate(tomo.sampling_history):
        sample_y, sample_x = np.where(mask)
        ratio = mask.sum() / (tomo.grid_size ** 2) * 100
        if i == 0:
            label = f"Round {i+1} (初始 {ratio:.1f}%)"
        else:
            label = f"Round {i+1} (+{ratio:.1f}%)"
        ax.scatter(tomo.X[0, sample_x], tomo.P[sample_y, 0], 
                  s=8, c=[colors[i]], alpha=0.8, label=label)
    
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(f"Sampling Distribution: {tomo.state_name}", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    filepath = f"{save_dir}/1_sampling_distribution.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_fidelity_curves(tomo, save_dir):
    """图2: 保真度曲线"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x_vals = [r*100 for r in tomo.history['sampling_ratio']]
    
    # F(重构 vs 实验) - 实线点
    ax.plot(x_vals, tomo.history['F_recon_vs_exp'], 'b-o', 
            linewidth=2, markersize=6, label='F(重构 vs 实验)')
    
    # F(重构 vs 理论) - 实线三角
    ax.plot(x_vals, tomo.history['F_recon_vs_ideal'], 'g-^', 
            linewidth=2, markersize=6, label='F(重构 vs 理论)')
    
    # F(实验 vs 理论) - 虚线
    ax.axhline(y=tomo.F_exp_vs_ideal, color='r', linestyle='--', 
               linewidth=2, label=f'F(实验 vs 理论) = {tomo.F_exp_vs_ideal:.4f}')
    
    # 目标线
    ax.axhline(y=tomo.F_threshold, color='gray', linestyle=':', 
               linewidth=1.5, label=f'目标 F = {tomo.F_threshold}')
    
    ax.set_xlabel("Sampling Rate (%)", fontsize=12)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title(f"Fidelity vs Sampling Rate: {tomo.state_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    filepath = f"{save_dir}/2_fidelity_curves.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_reconstruction(tomo, save_dir):
    """图3: 重构Wigner函数"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    vmax = max(np.max(np.abs(tomo.final_pred)), np.max(np.abs(tomo.exp_wigner)))
    cf = ax.contourf(tomo.X, tomo.P, tomo.final_pred, levels=50, 
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('W(x, p)', fontsize=11)
    
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(f"Reconstructed Wigner Function\nF = {tomo.final_F_exp:.4f}", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = f"{save_dir}/3_reconstruction.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_experimental_state(tomo, save_dir):
    """图4: 实验态Wigner函数"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    vmax = np.max(np.abs(tomo.exp_wigner))
    cf = ax.contourf(tomo.X, tomo.P, tomo.exp_wigner, levels=50, 
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('W(x, p)', fontsize=11)
    
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(f"Experimental Wigner Function: {tomo.state_name}\n"
                f"F(exp vs ideal) = {tomo.F_exp_vs_ideal:.4f}", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = f"{save_dir}/4_experimental_state.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_relative_error(tomo, save_dir):
    """图5: 相对误差百分比"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 计算相对误差 (避免除零)
    exp_abs = np.abs(tomo.exp_wigner) + 1e-10
    error = tomo.final_pred - tomo.exp_wigner
    relative_error = (error / exp_abs) * 100
    
    # 限制在合理范围
    relative_error = np.clip(relative_error, -100, 100)
    
    cf = ax.contourf(tomo.X, tomo.P, relative_error, levels=50, cmap='RdBu_r')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Relative Error (%)', fontsize=11)
    
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title("Relative Error: (W_recon - W_exp) / W_exp × 100%", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # 添加统计信息
    mean_err = np.mean(np.abs(relative_error))
    max_err = np.max(np.abs(relative_error))
    ax.text(0.02, 0.98, f"Mean |Error|: {mean_err:.2f}%\nMax |Error|: {max_err:.2f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = f"{save_dir}/5_relative_error.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_sampling_density(tomo, save_dir):
    """图6: 采样密度热力图"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 计算采样密度（使用高斯平滑）
    from scipy.ndimage import gaussian_filter
    density = tomo.sampling_mask.astype(float)
    density_smooth = gaussian_filter(density, sigma=2)
    
    cf = ax.imshow(density_smooth, extent=[tomo.X.min(), tomo.X.max(), 
                                           tomo.P.min(), tomo.P.max()],
                   origin='lower', cmap='hot', aspect='equal')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Sampling Density', fontsize=11)
    
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.set_title(f"Sampling Density Heatmap\nTotal: {tomo.sampling_mask.sum()} points "
                f"({tomo.final_ratio*100:.1f}%)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = f"{save_dir}/6_sampling_density.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()
