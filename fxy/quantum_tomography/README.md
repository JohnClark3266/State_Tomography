# 量子态层析项目 (Quantum Tomography)

基于主动学习的量子态Wigner函数稀疏层析系统。

## 模块结构

```
quantum_tomography/
├── 核心模块
│   ├── quantum_states.py     # 量子态生成 (Fock, 相干态, 猫态)
│   ├── neural_networks.py    # 20个CNN/ResNet模型池
│   ├── noise_model.py        # 实验噪声模拟
│   └── wigner_mle.py         # QuTiP Wigner函数和MLE重建
│
├── 采样模块 (新架构)
│   ├── sampling_manager.py   # 采样状态矩阵管理 (3, N)
│   ├── decision_maker.py     # 采样决策 (独立于执行)
│   └── fast_tomography.py    # 快速层析主逻辑
│
├── 旧版模块
│   └── active_learning.py    # 原始主动学习实现
│
├── 可视化
│   └── visualization.py      # 6张结果图生成
│
├── 入口
│   ├── main.py               # 原始入口
│   └── main_fast.py          # 快速版入口 (推荐)
│
└── 输出
    └── results/              # 结果图片
```

## 各模块功能

### 1. `quantum_states.py` - 量子态生成
- `fock_wigner(x, p, n)`: Fock态 |n⟩ 的Wigner函数
- `coherent_wigner(x, p, alpha)`: 相干态 |α⟩ 的Wigner函数
- `cat_wigner(x, p, alpha, parity)`: 猫态的Wigner函数
- `create_state()`: 统一接口创建量子态

### 2. `neural_networks.py` - 神经网络模型池
- 20个不同架构的CNN/ResNet/FC模型
- `build_model_pool()`: 构建模型池
- `select_committee()`: 随机选择委员会成员

### 3. `noise_model.py` - 噪声模拟
- `ExperimentalNoise`: 实验噪声类
  - 态畸变: 光子损耗、校准漂移
  - 测量噪声: 散粒噪声、读出噪声、暗计数
- `calibrate_noise_for_fidelity()`: 校准噪声以达到目标保真度
- `compute_fidelity()`: 计算重叠积分保真度

### 4. `sampling_manager.py` - 采样状态管理 (新)
- 管理 (3, N) 状态矩阵:
  - Row 0: 相空间坐标
  - Row 1: 采样状态 (0=未采, 1=待采, 2=已采)
  - Row 2: Wigner测量值
- `execute_sampling()`: 执行采样并填充值
- `fill_predictions()`: 填充预测值到未采样点

### 5. `decision_maker.py` - 采样决策 (新)
- 完全独立于采样执行
- 根据委员会不确定性 + 梯度 + 原点权重选择下一轮采样点
- `decide_next_samples()`: 决定下一轮采样点
- `decide_initial_samples()`: 决定初始采样点

### 6. `fast_tomography.py` - 快速层析 (新)
- 使用新的模块化架构
- 预训练后轻量微调 (比原版快10倍)
- 保真度可达 0.98+

### 7. `visualization.py` - 可视化
生成6张结果图:
1. 采样分布 (每轮不同颜色)
2. 保真度曲线
3. 重构Wigner函数
4. 实验态Wigner函数
5. 相对误差
6. 采样密度热力图

### 8. `wigner_mle.py` - Wigner函数和MLE
- 基于QuTiP的精确Wigner函数计算
- `wigner_MLE()`: 最大似然估计重建密度矩阵
- `compute_fidelity_qutip()`: 量子保真度计算

## 使用方法

### 快速开始 (推荐)
```bash
# 相干态，每轮增加10个采样点
python main_fast.py --state 2 --alpha 2.0 --samples 10

# Fock态 |3⟩
python main_fast.py --state 1 --n 3 --samples 10

# 猫态
python main_fast.py --state 3 --alpha 2.0 --samples 10
```

### 参数说明
- `--state`: 态类型 (1=Fock, 2=相干态, 3=猫态)
- `--alpha`: 相干态/猫态振幅
- `--n`: Fock态光子数
- `--samples`: 每轮增加的采样点数
- `--rounds`: 最大轮数
- `--pretrain`: 预训练epochs

## 结果

训练完成后，6张图保存在 `results/` 目录:
- `1_sampling_distribution.png`
- `2_fidelity_curves.png`
- `3_reconstruction.png`
- `4_experimental_state.png`
- `5_relative_error.png`
- `6_sampling_density.png`
