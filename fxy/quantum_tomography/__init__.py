"""
quantum_tomography - 量子态稀疏层析包

模块:
- quantum_states: 量子态Wigner函数生成 (Fock, Coherent, Cat)
- neural_networks: 20个神经网络模型池
- noise_model: 实验噪声模拟
- active_learning: 主动学习层析
- visualization: 6张可视化输出
"""

from .quantum_states import (
    fock_wigner, coherent_wigner, cat_wigner, 
    create_state, FOCK, COHERENT, CAT
)
from .neural_networks import build_model_pool, select_committee
from .noise_model import ExperimentalNoise, compute_fidelity, calibrate_noise_for_fidelity
from .active_learning import QuantumTomography
from .visualization import plot_all_results

__all__ = [
    'fock_wigner', 'coherent_wigner', 'cat_wigner', 'create_state',
    'FOCK', 'COHERENT', 'CAT',
    'build_model_pool', 'select_committee',
    'ExperimentalNoise', 'compute_fidelity', 'calibrate_noise_for_fidelity',
    'QuantumTomography',
    'plot_all_results',
]
