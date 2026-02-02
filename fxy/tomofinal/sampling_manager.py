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
                 matlab_bridge: Optional['MatlabBridge'] = None,
                 ground_truth_wigner: Optional[np.ndarray] = None):
        """初始化采样管理器"""
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.x_range = x_range
        self.matlab_bridge = matlab_bridge
        self.ground_truth_wigner = ground_truth_wigner
        
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
        """返回用于 MATLAB 的完整状态矩阵"""
        state = np.array(list(self.state_matrix[1, :]))
        wigner = np.array(list(self.state_matrix[2, :]))
        coords = self.coordinates
        return state, wigner, coords
    
    def update_from_matlab_matrix(self, new_state: np.ndarray, new_wigner: np.ndarray):
        """从 MATLAB 矩阵更新本地状态"""
        for i in range(self.n_points):
            # 防止状态回退 (已采样 -> 未采样)
            if self.state_matrix[1, i] == self.STATE_SAMPLED:
                continue
                
            if new_state[i] == self.STATE_SAMPLED:
                self.state_matrix[1, i] = self.STATE_SAMPLED
                self.state_matrix[2, i] = new_wigner[i]
    
    def execute_sampling_in_matlab(self, matlab_script_path: str = None) -> bool:
        """
        执行 MATLAB 采样流程
        
        1. 发送当前状态到 MATLAB
        2. 运行 MATLAB 脚本 (模拟实验测量)
        3. 接收 MATLAB 更新后的状态
        """
        # 准备数据
        py_matrix_state, py_matrix_wigner, coords = self.get_full_state_matrix_for_matlab()
        
        # 简单统计
        n_to_sample = np.sum(py_matrix_state == self.STATE_TO_SAMPLE)
        if n_to_sample == 0:
            print("  没有需要采样的点")
            return True
            
        print(f"  [采样] 准备采样 {n_to_sample} 个点...")
        
        # === Mock Mode (无 MATLAB) ===
        if (self.matlab_bridge is None or not self.matlab_bridge.is_connected) and self.ground_truth_wigner is not None:
             print("  [Mock] 使用本地 Ground Truth 进行模拟采样...")
             
             # 获取待采样索引
             indices = np.where(py_matrix_state == self.STATE_TO_SAMPLE)[0]
             
             # 模拟测量 (Ground Truth + 噪声)
             true_values = self.ground_truth_wigner.flatten()[indices]
             noise = np.random.normal(0, 0.006, size=true_values.shape) # 使用 config default noise
             measured_values = true_values + noise
             
             # 更新状态
             self.state_matrix[1, indices] = self.STATE_SAMPLED
             self.state_matrix[2, indices] = measured_values
             
             # Add to history
             if len(indices) > 0:
                 self.sampling_history.append(indices.copy())
             
             print(f"  ✓ Mock采样完成，新增 {n_to_sample} 点")
             return True

        # === MATLAB Mode ===
        if not self.matlab_bridge.is_connected:
            print("✗ 无法采样: 未连接 MATLAB 且无 Ground Truth")
            return False

        print(f"  [Python->MATLAB] 发送 py_matrix_state, py_matrix_wigner (1×{self.n_points})...")
        self.matlab_bridge.send('py_matrix_state', py_matrix_state)
        self.matlab_bridge.send('py_matrix_wigner', py_matrix_wigner)
        self.matlab_bridge.send('py_matrix_coords', coords.reshape(1, -1))
        
        # 运行 MATLAB 脚本
        if matlab_script_path:
            print(f"  [MATLAB] 运行脚本: {matlab_script_path}")
            self.matlab_bridge.eval(f"run('{matlab_script_path}')")
        else:
            print("  ⚠ 未指定 MATLAB 脚本路径")
            return False
        
        # 接收结果
        print(f"  [MATLAB->Python] 接收 mat_matrix_state, mat_matrix_wigner (1×{self.n_points})...")
        mat_matrix_state = self.matlab_bridge.receive('mat_matrix_state')
        mat_matrix_wigner = self.matlab_bridge.receive('mat_matrix_wigner')
        
        if mat_matrix_state is None or mat_matrix_wigner is None:
            print("⚠ MATLAB 返回失败 (检查 mat_matrix_state/wigner)")
            return False
        
        # 解析数据
        new_state = np.array(mat_matrix_state).flatten()
        new_wigner = np.array(mat_matrix_wigner).flatten()
        
        # Identify newly sampled points for history
        old_state_arr = self.get_state()
        newly_sampled_indices = np.where((old_state_arr == self.STATE_TO_SAMPLE) & (new_state == self.STATE_SAMPLED))[0]
        if len(newly_sampled_indices) > 0:
            self.sampling_history.append(newly_sampled_indices.copy())
            
        self.update_from_matlab_matrix(new_state, new_wigner)
        print(f"  ✓ 采样完成，新增 {n_to_sample} 点")
        return True
    
    def get_wigner_target_from_matlab(self) -> Optional[np.ndarray]:
        """
        从 MATLAB workspace 读取 Wigner_target
        作为实验态的基准 (用于保真度计算)
        
        返回:
            wigner: Wigner 函数 (grid_size, grid_size)
        """
        # === Mock Mode ===
        if self.ground_truth_wigner is not None:
             # 如果无法连接 MATLAB，优先返回本地 GT
             if self.matlab_bridge is None or not self.matlab_bridge.is_connected:
                 print("  [Mock] 返回本地 Ground Truth Wigner")
                 return self.ground_truth_wigner
        
        if self.matlab_bridge is None or not self.matlab_bridge.is_connected:
            return None
        
        try:
            # 尝试获取 Wigner_target
            wigner = self.matlab_bridge.receive('Wigner_target', silent=True)
            
            if wigner is not None:
                wigner_arr = np.array(wigner)
                print(f"  ✓ 从 MATLAB 获取 Wigner_target {wigner_arr.shape}")
                
                # 确保形状正确
                if wigner_arr.shape != (self.grid_size, self.grid_size):
                    print(f"  ⚠ 警告: MATLAB Wigner 形状 {wigner_arr.shape} 不匹配 ({self.grid_size}, {self.grid_size})")
                    # 尝试 resize 或 reshape? 暂时不处理，让它报错或保持原样
                
                return wigner_arr
        except Exception as e:
            print(f"  ⚠ 获取 Wigner_target 失败: {e}")
        
        return None
    
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
