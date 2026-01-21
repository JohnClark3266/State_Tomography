"""
tomography - GKP态稀疏量子态层析包

模块说明:
- gkp_state: GKP态Wigner函数生成
- noise_model: 实验噪声模拟
- sparse_sampling: 稀疏采样与数据生成
- cnn_models: CNN网络架构
- fidelity: 保真度计算
- active_learning: 主动学习层析
"""

from .gkp_state import gkp_wigner, create_gkp_grid
from .noise_model import ExperimentalNoise, calibrate_noise_for_fidelity
from .sparse_sampling import create_sparse_input, generate_random_mask, generate_training_data
from .cnn_models import CNN1, CNN2, CNN3, CNN4, CNN5, build_cnn_committee
from .fidelity import compute_fidelity
from .active_learning import ActiveSparseTomography

__all__ = [
    'gkp_wigner',
    'create_gkp_grid',
    'ExperimentalNoise',
    'calibrate_noise_for_fidelity',
    'create_sparse_input',
    'generate_random_mask',
    'generate_training_data',
    'CNN1', 'CNN2', 'CNN3', 'CNN4', 'CNN5',
    'build_cnn_committee',
    'compute_fidelity',
    'ActiveSparseTomography',
]
