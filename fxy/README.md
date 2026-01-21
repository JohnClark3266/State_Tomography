# GKP态稀疏量子态层析

基于主动学习和卷积神经网络的GKP（Gottesman-Kitaev-Preskill）量子态相空间层析模拟。

## 项目简介

本项目实现了一种高效的量子态层析方法，通过稀疏采样和主动学习策略，能够以远低于传统方法的采样率（~20%）重建GKP量子态的Wigner函数，同时达到高保真度（>98%）。

### 核心特性

- **稀疏层析**: 仅需部分相空间采样点即可重建完整Wigner函数
- **主动学习**: 基于不确定性的采样点选择策略
- **CNN委员会**: 5种不同架构的神经网络集成，提高鲁棒性
- **实验噪声模拟**: 模拟真实量子光学实验中的各种噪声
- **三保真度评估**: F1/F2/F3 分别评估实验态、重建态与理论态的关系

## 目录结构

```
fxy/
├── tomography/              # 核心层析模块
│   ├── __init__.py          # 包初始化
│   ├── main.py              # 主入口
│   ├── gkp_state.py         # GKP态Wigner函数生成
│   ├── noise_model.py       # 实验噪声模拟
│   ├── sparse_sampling.py   # 稀疏采样与数据生成
│   ├── cnn_models.py        # 5种CNN架构
│   ├── fidelity.py          # 保真度计算
│   ├── active_learning.py   # 主动学习层析核心类
│   ├── gkp_wigner.py        # 独立的Wigner函数可视化
│   └── results/             # 输出结果
│       ├── experimental_tomography_results.png
│       └── committee_comparison.png
├── ppt/                     # 演示文稿
│   ├── presentation.html    # HTML演示
│   └── images/              # 演示图片
└── README.md                # 本文档
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- SciPy
- Matplotlib

### 安装依赖

```bash
pip install torch numpy scipy matplotlib
```

### 运行示例

```bash
cd tomography
python main.py
```

## 模块说明

### gkp_state.py - GKP态生成

提供GKP量子态的Wigner函数计算：

```python
from tomography import create_gkp_grid

X, P, W = create_gkp_grid(grid_size=64, delta=0.3)
```

### noise_model.py - 噪声模拟

模拟实验中的各种噪声源：

- **态畸变**: 光子损耗（高斯扩散）、校准漂移
- **测量噪声**: 散粒噪声、读出噪声、暗计数、背景噪声

```python
from tomography import ExperimentalNoise

noise = ExperimentalNoise(
    detection_efficiency=0.85,  # 探测效率
    shot_noise_scale=0.05,      # 散粒噪声
)
```

### cnn_models.py - CNN架构

包含5种不同的卷积神经网络：

| 模型 | 架构 | 激活函数 |
|------|------|----------|
| CNN1 | 3层编码-解码 | ReLU |
| CNN2 | 4层串联 | Tanh |
| CNN3 | 宽通道 | LeakyReLU |
| CNN4 | ResNet风格 | ReLU+跳跃连接 |
| CNN5 | U-Net风格 | ReLU+编解码 |

### active_learning.py - 主动学习

核心层析类，实现完整的训练流程：

```python
from tomography import ActiveSparseTomography

tomo = ActiveSparseTomography(
    grid_size=64,
    target_experimental_fidelity=0.95,
    F2_threshold=0.99
)
tomo.run()
tomo.plot_results()
```

## 三保真度指标

| 指标 | 定义 | 意义 |
|------|------|------|
| F₁ | Exp vs Ideal | 实验态与理论态的差距（由噪声决定） |
| F₂ | Recon vs Exp | 重建态对实验态的还原度（**优化目标**） |
| F₃ | Recon vs Ideal | 重建态与理论态的相似度 |

**关键洞察**: F₂ > F₃ 表明网络正确学习了实验中的非理想特征。

## 输出结果

运行后生成的结果图包括：

1. **experimental_tomography_results.png**: 主结果图
   - 实验态、理想态、重建态对比
   - 采样点分布
   - 三保真度收敛曲线
   - 不确定性热图

2. **committee_comparison.png**: 委员会成员对比
   - 5个CNN的单独预测
   - 集成平均效果

## 参考文献

- Gottesman, D., Kitaev, A., & Preskill, J. (2001). Encoding a qubit in an oscillator. Physical Review A.
- GKP态的Wigner表示和量子纠错

## 许可证

MIT License
