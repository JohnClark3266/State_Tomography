# 简化版量子态层析

基于 `Wigner_MLE_fidelity.py` 方法的量子态层析，使用 QuTiP 生成 Wigner 函数。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序 (QuTiP Wigner + MLE 保真度) |
| `neural_networks.py` | 20 个神经网络模型 |
| `sampling_manager.py` | MATLAB 采样管理 |
| `decision_maker.py` | 主动学习决策 |
| `matlab_bridge.py` | MATLAB Engine 接口 |
| `visualization.py` | 结果可视化 |

## 依赖

```bash
pip install numpy torch scipy matplotlib qutip cvxpy
```

## 使用方法

```bash
# 确保 MATLAB 已运行并执行了 matlab.engine.shareEngine

python main.py --state 2 --alpha 2.0 --threshold 0.99
python main.py --state 1 --n 3 --threshold 0.99  # Fock态
```

## 核心流程

1. **QuTiP 生成态**: 使用 `qutip.coherent()` 或 `qutip.basis()` 创建量子态
2. **MLE 初始化**: 从 Wigner 函数重建密度矩阵
3. **预训练**: 用完整 4096 点训练神经网络
4. **主动学习**: 基于委员会方差选择采样点
5. **MATLAB 采样**: 获取真实 Wigner 值
6. **保真度计算**: 使用 `qutip.fidelity()` 比较密度矩阵
