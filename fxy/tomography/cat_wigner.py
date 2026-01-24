"""
猫态Wigner函数可视化和无噪声层析测试
"""

import numpy as np
import matplotlib.pyplot as plt
from cat_state import create_cat_grid, cat_wigner

def plot_cat_wigner(alpha=2.5, parity='even', grid_size=100, save_path='results/cat_wigner.png'):
    """绘制猫态的Wigner函数"""
    
    X, P, W = create_cat_grid(grid_size=grid_size, x_range=(-5, 5), alpha=alpha, parity=parity)
    
    parity_cn = "偶" if parity == 'even' else "奇"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制等高线填充图
    levels = np.linspace(W.min(), W.max(), 50)
    cf = ax.contourf(X, P, W, levels=levels, cmap='RdBu_r')
    
    # 添加等高线
    cs = ax.contour(X, P, W, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    
    # 颜色条
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('W(x, p)', fontsize=12)
    
    ax.set_xlabel('x (position)', fontsize=12)
    ax.set_ylabel('p (momentum)', fontsize=12)
    ax.set_title(f'{parity_cn}猫态 Wigner Function (α={alpha})', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存Wigner函数图到 {save_path}")
    plt.close()
    
    return X, P, W

if __name__ == "__main__":
    # 绘制偶猫态Wigner函数
    plot_cat_wigner(alpha=2.5, parity='even', save_path='results/cat_wigner_even_alpha2.5.png')
    
    # 绘制奇猫态Wigner函数
    plot_cat_wigner(alpha=2.5, parity='odd', save_path='results/cat_wigner_odd_alpha2.5.png')
    
    print("\n猫态Wigner函数可视化完成!")
