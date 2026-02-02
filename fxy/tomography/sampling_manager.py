"""
采样状态管理模块 (MATLAB 专用)

管理采样状态矩阵并与 MATLAB 通信。

状态定义:
- 0: 未采样
- 1: 待采样 (下轮采)  
- 2: 已采样
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matlab_bridge import MatlabBridge


class SamplingManager:
    """采样状态管理器 (MATLAB 采样)"""
    
    STATE_UNSAMPLED = 0
    STATE_TO_SAMPLE = 1
    STATE_SAMPLED = 2
    
    def __init__(self, grid_size: int = 64, x_range: Tuple[float, float] = (-5, 5),
                 matlab_bridge: Optional['MatlabBridge'] = None):
        """初始化采样管理器"""
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.x_range = x_range
        self.matlab_bridge = matlab_bridge
        
        # 相空间坐标网格
        x = np.linspace(x_range[0], x_range[1], grid_size)
        X, P = np.meshgrid(x, x)
        self.X_flat = X.flatten()
        self.P_flat = P.flatten()
        self.coordinates = self.X_flat + 1j * self.P_flat
        
        # 状态矩阵 (3, N): [坐标, 状态, Wigner值]
        self.state_matrix = np.empty((3, self.n_points), dtype=object)
        self.state_matrix[0, :] = self.coordinates
        self.state_matrix[1, :] = self.STATE_UNSAMPLED
        self.state_matrix[2, :] = 0.0
        
        self.sampling_history = []
    
    # =============================================
    # 核心方法 (MATLAB 采样必需)
    # =============================================
    
    def get_state(self) -> np.ndarray:
        """返回状态数组 (N,)"""
        return np.array(list(self.state_matrix[1, :]), dtype=np.float64)
    
    def get_wigner_2d(self) -> np.ndarray:
        """返回 2D Wigner 函数"""
        wigner = np.array(list(self.state_matrix[2, :]), dtype=np.float64)
        return wigner.reshape(self.grid_size, self.grid_size)
    
    def get_mask_2d(self) -> np.ndarray:
        """返回 2D 采样掩码"""
        state = self.get_state()
        return (state == self.STATE_SAMPLED).reshape(self.grid_size, self.grid_size)
    
    def set_points_to_sample(self, indices: np.ndarray):
        """设置待采样点"""
        for idx in indices:
            if self.state_matrix[1, idx] == self.STATE_UNSAMPLED:
                self.state_matrix[1, idx] = self.STATE_TO_SAMPLE
    
    def get_sampled_count(self) -> int:
        """返回已采样点数"""
        return int(np.sum(self.get_state() == self.STATE_SAMPLED))
    
    def get_sampling_ratio(self) -> float:
        """返回采样率"""
        return self.get_sampled_count() / self.n_points
    
    def get_sparse_input_for_nn(self) -> np.ndarray:
        """返回神经网络输入 (2, grid, grid)"""
        wigner_2d = self.get_wigner_2d()
        mask_2d = self.get_mask_2d().astype(np.float32)
        sparse_wigner = np.where(mask_2d, wigner_2d, 0.0)
        return np.stack([sparse_wigner, mask_2d], axis=0).astype(np.float32)
    
    def fill_predictions(self, predictions: np.ndarray):
        """填充未采样点的预测值"""
        if predictions.ndim == 2:
            predictions = predictions.flatten()
        for i in range(self.n_points):
            if self.state_matrix[1, i] == self.STATE_UNSAMPLED:
                self.state_matrix[2, i] = predictions[i]
    
    # =============================================
    # MATLAB 采样接口
    # =============================================
    
    def get_full_state_matrix_for_matlab(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取完整状态矩阵用于发送给 MATLAB"""
        coordinates = self.coordinates.copy()
        state = self.get_state()
        wigner = np.array(list(self.state_matrix[2, :]), dtype=np.float64)
        return coordinates, state, wigner
    
    def update_from_matlab_matrix(self, state: np.ndarray, wigner: np.ndarray):
        """从 MATLAB 返回数据更新本地状态"""
        state = state.flatten()
        wigner = wigner.flatten()
        
        old_state = self.get_state()
        new_sampled = (old_state == self.STATE_TO_SAMPLE) & (state == self.STATE_SAMPLED)
        new_indices = np.where(new_sampled)[0]
        
        for i in range(self.n_points):
            self.state_matrix[1, i] = state[i]
            self.state_matrix[2, i] = wigner[i]
        
        if len(new_indices) > 0:
            self.sampling_history.append(new_indices.copy())
    
    def execute_sampling_matlab(self, state_type: int = 2, noise_std: float = 0.02,
                                  **state_params) -> bool:
        """
        使用 MATLAB 执行采样 (调用 run('Active_learning_function.m'))
        """
        if self.matlab_bridge is None or not self.matlab_bridge.is_connected:
            print("⚠ MATLAB 未连接")
            return False
        
        # 切换 MATLAB 工作目录到当前脚本所在目录
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.matlab_bridge.eval(f"cd('{current_dir}')")
        
        coords, state, wigner = self.get_full_state_matrix_for_matlab()
        n_to_sample = int(np.sum(state == self.STATE_TO_SAMPLE))
        if n_to_sample == 0:
            return True
            
        print(f"  [MATLAB] 调用 Active_learning_function.m (待采样: {n_to_sample})...")
        
        # 发送数据 (匹配 Active_learning_function.m 的变量名)
        self.matlab_bridge.send('py_matrix_coords', coords.reshape(1, -1))
        self.matlab_bridge.send('py_matrix_state', state.reshape(1, -1))
        self.matlab_bridge.send('py_matrix_wigner', wigner.reshape(1, -1))
        
        # 发送状态类型参数
        self.matlab_bridge.send('state', np.array([float(state_type)]))
        
        # 执行脚本
        self.matlab_bridge.eval("run('Active_learning_function.m')")
        
        # 接收结果
        new_state = self.matlab_bridge.receive('mat_matrix_state')
        new_wigner = self.matlab_bridge.receive('mat_matrix_wigner')
        
        if new_state is None or new_wigner is None:
            print("⚠ MATLAB 返回失败 (mat_matrix_state 或 mat_matrix_wigner 未找到)")
            return False
            
        self.update_from_matlab_matrix(new_state, new_wigner)
        print(f"  ✓ 采样完成")
        return True
    
    # =============================================
    # Python 备用采样 (已注释)
    # =============================================
    
    # def execute_sampling(self, wigner_source: np.ndarray, noise_model=None):
    #     """
    #     Python 备用采样：将状态=1的点采样并填充Wigner值
    #     
    #     参数:
    #         wigner_source: 源Wigner函数 (N,) 或 (grid, grid)
    #         noise_model: 可选的噪声模型
    #     """
    #     if wigner_source.ndim == 2:
    #         wigner_source = wigner_source.flatten()
    #     
    #     state = self.get_state()
    #     to_sample = state == self.STATE_TO_SAMPLE
    #     indices = np.where(to_sample)[0]
    #     
    #     if len(indices) == 0:
    #         return
    #     
    #     measured_values = wigner_source[indices].copy()
    #     if noise_model is not None:
    #         noise = np.random.normal(0, 0.02, size=measured_values.shape)
    #         measured_values += noise
    #     
    #     self.state_matrix[2, indices] = measured_values
    #     self.state_matrix[1, indices] = self.STATE_SAMPLED
    #     self.sampling_history.append(indices.copy())
    
    # def get_coordinates(self) -> np.ndarray:
    #     """返回完整复数坐标 (N,)"""
    #     return self.coordinates.copy()
    
    # def get_coordinates_2d(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """返回2D网格坐标"""
    #     X = self.X_flat.reshape(self.grid_size, self.grid_size)
    #     P = self.P_flat.reshape(self.grid_size, self.grid_size)
    #     return X, P
    
    # def get_wigner_values(self) -> np.ndarray:
    #     """返回Wigner值向量 (N,)"""
    #     return np.array(list(self.state_matrix[2, :]), dtype=np.float64)
    
    # def reset(self):
    #     """重置所有状态"""
    #     self.state_matrix[1, :] = self.STATE_UNSAMPLED
    #     self.state_matrix[2, :] = 0.0
    #     self.sampling_history = []
