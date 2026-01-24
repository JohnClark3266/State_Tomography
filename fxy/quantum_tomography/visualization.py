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
    """图1: 采样分布（每轮新增点用不同颜色）"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 半透明理论态衬底
    vmax = np.max(np.abs(tomo.ideal_wigner))
    ax.contourf(tomo.X, tomo.P, tomo.ideal_wigner, levels=40, 
                cmap='RdBu_r', alpha=0.3, vmin=-vmax, vmax=vmax)
    
    # 为每轮采样点分配不同颜色
    n_rounds = len(tomo.sampling_history)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(n_rounds, 1)))
    
    # 记录累积的点，以便只画每轮新增的点
    prev_mask = np.zeros((tomo.grid_size, tomo.grid_size), dtype=bool)
    
    for i, mask in enumerate(tomo.sampling_history):
        # 只取这一轮新增的点 (当前mask减去之前的累积mask)
        new_points_mask = mask & (~prev_mask)
        
        # 获取新增采样点的行列索引
        row_indices, col_indices = np.where(new_points_mask)
        
        if len(row_indices) == 0:
            prev_mask = mask.copy()
            continue
        
        # 将索引转换为实际坐标值
        x_coords = tomo.X[0, col_indices]
        p_coords = tomo.P[row_indices, 0]
        
        n_new = new_points_mask.sum()
        total = mask.sum()
        
        if i == 0:
            label = f"Round {i+1}: {n_new} pts (Initial)"
        else:
            label = f"Round {i+1}: +{n_new} pts"
        
        ax.scatter(x_coords, p_coords, s=15, c=[colors[i]], 
                   alpha=0.9, label=label, edgecolors='none')
        
        prev_mask = mask.copy()
    
    ax.set_xlabel("x (Re direction)", fontsize=12)
    ax.set_ylabel("p (Im direction)", fontsize=12)
    total_sampled = tomo.sampling_mask.sum()
    ratio = total_sampled / (tomo.grid_size ** 2) * 100
    ax.set_title(f"Sampling Distribution: {tomo.state_name}\n"
                 f"Total: {total_sampled} points ({ratio:.1f}%)", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # 图例放在右侧外面，避免遮挡
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    plt.tight_layout()
    filepath = f"{save_dir}/1_sampling_distribution.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    """图3: 重构Wigner函数 (Wigner=0显示为白色)"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 自定义colormap: 负值蓝色，零值白色，正值红色
    colors = [(0.0, 'darkblue'), (0.25, 'blue'), (0.5, 'white'), 
              (0.75, 'red'), (1.0, 'darkred')]
    cmap = LinearSegmentedColormap.from_list('wigner_cmap', colors)
    
    vmax = max(np.max(np.abs(tomo.final_pred)), np.max(np.abs(tomo.exp_wigner)))
    cf = ax.contourf(tomo.X, tomo.P, tomo.final_pred, levels=50, 
                     cmap=cmap, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('W(x, p)', fontsize=11)
    
    ax.set_xlabel("Re(alpha)", fontsize=12)
    ax.set_ylabel("Im(alpha)", fontsize=12)
    ax.set_title(f"Reconstructed Wigner Function\\nF = {tomo.final_F_exp:.4f}", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = f"{save_dir}/3_reconstruction.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_experimental_state(tomo, save_dir):
    """图4: 实验态Wigner函数 (Wigner=0显示为白色)"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 自定义colormap: 负值蓝色，零值白色，正值红色
    colors = [(0.0, 'darkblue'), (0.25, 'blue'), (0.5, 'white'), 
              (0.75, 'red'), (1.0, 'darkred')]
    cmap = LinearSegmentedColormap.from_list('wigner_cmap', colors)
    
    vmax = np.max(np.abs(tomo.exp_wigner))
    cf = ax.contourf(tomo.X, tomo.P, tomo.exp_wigner, levels=50, 
                     cmap=cmap, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('W(x, p)', fontsize=11)
    
    ax.set_xlabel("Re(alpha)", fontsize=12)
    ax.set_ylabel("Im(alpha)", fontsize=12)
    ax.set_title(f"Experimental Wigner Function: {tomo.state_name}\\n"
                f"F(exp vs ideal) = {tomo.F_exp_vs_ideal:.4f}", 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = f"{save_dir}/4_experimental_state.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()


def plot_relative_error(tomo, save_dir):
    """图5: 绝对误差 (比相对误差更有意义)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 计算误差
    error = tomo.final_pred - tomo.exp_wigner
    abs_error = np.abs(error)
    
    # 左图: 绝对误差
    ax1 = axes[0]
    vmax_err = np.max(abs_error)
    cf1 = ax1.contourf(tomo.X, tomo.P, abs_error, levels=50, cmap='hot')
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label('Absolute Error', fontsize=11)
    
    ax1.set_xlabel("x (Re direction)", fontsize=12)
    ax1.set_ylabel("p (Im direction)", fontsize=12)
    ax1.set_title("Absolute Error: |W_recon - W_target|", fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # 添加统计信息
    mean_abs = np.mean(abs_error)
    max_abs = np.max(abs_error)
    ax1.text(0.02, 0.98, f"Mean: {mean_abs:.4f}\nMax: {max_abs:.4f}",
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 右图: 有符号误差 (正/负)
    ax2 = axes[1]
    vmax = max(np.max(error), -np.min(error))
    cf2 = ax2.contourf(tomo.X, tomo.P, error, levels=50, cmap='RdBu_r', 
                        vmin=-vmax, vmax=vmax)
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label('Signed Error', fontsize=11)
    
    ax2.set_xlabel("x (Re direction)", fontsize=12)
    ax2.set_ylabel("p (Im direction)", fontsize=12)
    ax2.set_title("Signed Error: W_recon - W_target", fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    
    # RMS 误差
    rms = np.sqrt(np.mean(error**2))
    ax2.text(0.02, 0.98, f"RMS: {rms:.4f}",
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = f"{save_dir}/5_relative_error.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    
    ax.set_xlabel("Re(alpha)", fontsize=12)
    ax.set_ylabel("Im(alpha)", fontsize=12)
    ax.set_title(f"Sampling Density Heatmap\nTotal: {tomo.sampling_mask.sum()} points "
                f"({tomo.final_ratio*100:.1f}%)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = f"{save_dir}/6_sampling_density.png"
    plt.savefig(filepath, dpi=300)
    print(f"保存: {filepath}")
    plt.close()
