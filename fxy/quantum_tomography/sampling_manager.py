"""
采样状态管理模块

管理一个 (3, N) 矩阵:
- Row 0: 复数坐标 (相空间位置)
- Row 1: 采样状态 (0=未采/不采, 1=未采/下轮采, 2=已采)
- Row 2: Wigner值 (已采=测量值, 未采=0)

此模块负责:
- 根据状态=1的点执行采样
- 填充Wigner测量值
- 不负责决策下一轮采哪些点
"""

import numpy as np
from typing import Tuple, Optional


class SamplingManager:
    """采样状态管理器"""
    
    # 采样状态常量
    STATE_UNSAMPLED = 0      # 未采样，下轮不采
    STATE_TO_SAMPLE = 1      # 未采样，下轮采
    STATE_SAMPLED = 2        # 已采样
    
    def __init__(self, grid_size: int = 64, x_range: Tuple[float, float] = (-5, 5)):
        """
        初始化采样管理器
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            x_range: 相空间坐标范围
        """
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.x_range = x_range
        
        # 生成相空间坐标网格
        x = np.linspace(x_range[0], x_range[1], grid_size)
        p = np.linspace(x_range[0], x_range[1], grid_size)
        X, P = np.meshgrid(x, p)
        
        # 展平为1D数组
        self.X_flat = X.flatten()
        self.P_flat = P.flatten()
        
        # 复数坐标 (alpha = x + i*p)
        self.coordinates = self.X_flat + 1j * self.P_flat
        
        # 初始化状态矩阵 (3, N)
        # Row 0: 复数坐标 (完整复数)
        # Row 1: 状态 (0/1/2)
        # Row 2: Wigner值
        # 使用对象数组以支持混合类型
        self.state_matrix = np.empty((3, self.n_points), dtype=object)
        self.state_matrix[0, :] = self.coordinates  # 存完整复数
        self.state_matrix[1, :] = self.STATE_UNSAMPLED  # 初始全为0
        self.state_matrix[2, :] = 0.0  # Wigner值初始为0
        
        # 同时保留数值类型的状态和Wigner值数组用于计算
        self._state_array = np.zeros(self.n_points, dtype=np.float64)
        self._wigner_array = np.zeros(self.n_points, dtype=np.float64)
        
        # 历史记录
        self.sampling_history = []
    
    def get_coordinates(self) -> np.ndarray:
        """返回完整复数坐标 (N,)"""
        return self.coordinates.copy()
    
    def get_coordinates_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回2D网格坐标"""
        X = self.X_flat.reshape(self.grid_size, self.grid_size)
        P = self.P_flat.reshape(self.grid_size, self.grid_size)
        return X, P
    
    def get_state(self) -> np.ndarray:
        """返回当前状态向量 (N,) - 数值类型"""
        return np.array(list(self.state_matrix[1, :]), dtype=np.float64)
    
    def get_wigner_values(self) -> np.ndarray:
        """返回当前Wigner值向量 (N,) - 数值类型"""
        return np.array(list(self.state_matrix[2, :]), dtype=np.float64)
    
    def get_wigner_2d(self) -> np.ndarray:
        """返回2D Wigner函数 - 数值类型"""
        wigner = np.array(list(self.state_matrix[2, :]), dtype=np.float64)
        return wigner.reshape(self.grid_size, self.grid_size)
    
    def get_mask_2d(self) -> np.ndarray:
        """返回2D采样掩码 (已采样=True)"""
        state = np.array(list(self.state_matrix[1, :]), dtype=np.float64)
        return (state == self.STATE_SAMPLED).reshape(
            self.grid_size, self.grid_size
        )
    
    def set_points_to_sample(self, indices: np.ndarray):
        """
        设置下一轮要采样的点 (状态改为1)
        
        参数:
            indices: 要采样的点的索引数组
        """
        # 只能设置未采样的点
        for idx in indices:
            if self.state_matrix[1, idx] == self.STATE_UNSAMPLED:
                self.state_matrix[1, idx] = self.STATE_TO_SAMPLE
    
    def execute_sampling(self, wigner_source: np.ndarray, noise_model=None):
        """
        执行采样：将状态=1的点采样并填充Wigner值
        
        参数:
            wigner_source: 源Wigner函数 (N,) 或 (grid, grid)
            noise_model: 可选的噪声模型
        """
        if wigner_source.ndim == 2:
            wigner_source = wigner_source.flatten()
        
        # 找到状态=1的点
        state = np.array(list(self.state_matrix[1, :]), dtype=np.float64)
        to_sample = state == self.STATE_TO_SAMPLE
        indices = np.where(to_sample)[0]
        
        if len(indices) == 0:
            return
        
        # 获取测量值 (可选加噪声)
        measured_values = wigner_source[indices].copy()
        if noise_model is not None:
            # 简化噪声：加高斯噪声
            noise = np.random.normal(0, 0.02, size=measured_values.shape)
            measured_values += noise
        
        # 填充Wigner值
        self.state_matrix[2, indices] = measured_values
        
        # 更新状态: 1 -> 2
        self.state_matrix[1, indices] = self.STATE_SAMPLED
        
        # 记录本轮采样
        self.sampling_history.append(indices.copy())
    
    def fill_predictions(self, predictions: np.ndarray):
        """
        填充未采样点的预测值 (状态保持0不变)
        
        参数:
            predictions: 预测的Wigner值 (N,) 或 (grid, grid)
        """
        if predictions.ndim == 2:
            predictions = predictions.flatten()
        
        # 只填充状态=0的点
        for i in range(self.n_points):
            if self.state_matrix[1, i] == self.STATE_UNSAMPLED:
                self.state_matrix[2, i] = predictions[i]
    
    def get_sampled_count(self) -> int:
        """返回已采样点数"""
        state = np.array(list(self.state_matrix[1, :]), dtype=np.float64)
        return int(np.sum(state == self.STATE_SAMPLED))
    
    def get_sampling_ratio(self) -> float:
        """返回采样率"""
        return self.get_sampled_count() / self.n_points
    
    def get_sparse_input_for_nn(self) -> np.ndarray:
        """
        返回神经网络输入格式: (2, grid, grid)
        Channel 0: 稀疏Wigner值 (未采样=0)
        Channel 1: 采样掩码
        """
        wigner_2d = self.get_wigner_2d()
        mask_2d = self.get_mask_2d().astype(np.float32)
        
        # 未采样点的Wigner值清零
        sparse_wigner = np.where(mask_2d, wigner_2d, 0.0)
        
        return np.stack([sparse_wigner, mask_2d], axis=0).astype(np.float32)
    
    def reset(self):
        """重置所有状态"""
        self.state_matrix[1, :] = self.STATE_UNSAMPLED
        self.state_matrix[2, :] = 0.0
        self.sampling_history = []
