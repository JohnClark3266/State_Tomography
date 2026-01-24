"""
采样状态管理模块 (支持 MATLAB 集成)

管理一个 (3, N) 矩阵:
- Row 0: 复数坐标 (相空间位置)
- Row 1: 采样状态 (0=未采/不采, 1=未采/下轮采, 2=已采)
- Row 2: Wigner值 (已采=测量值, 未采=0)

此模块负责:
- 根据状态=1的点执行采样
- 填充Wigner测量值
- 支持 MATLAB Engine 进行实际采样
- 不负责决策下一轮采哪些点
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matlab_bridge import MatlabBridge


class SamplingManager:
    """采样状态管理器"""
    
    # 采样状态常量
    STATE_UNSAMPLED = 0      # 未采样，下轮不采
    STATE_TO_SAMPLE = 1      # 未采样，下轮采
    STATE_SAMPLED = 2        # 已采样
    
    def __init__(self, grid_size: int = 64, x_range: Tuple[float, float] = (-5, 5),
                 matlab_bridge: Optional['MatlabBridge'] = None):
        """
        初始化采样管理器
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            x_range: 相空间坐标范围
            matlab_bridge: 可选的 MATLAB 桥接器 (用于真实采样)
        """
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.x_range = x_range
        self.matlab_bridge = matlab_bridge
        
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
    def set_matlab_bridge(self, bridge: 'MatlabBridge'):
        """设置 MATLAB 桥接器"""
        self.matlab_bridge = bridge
    
    def get_full_state_matrix_for_matlab(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取完整的状态矩阵 (3, N) 用于发送给 MATLAB
        
        返回三个分离的数组 (因为 MATLAB 不支持复数和实数混合的矩阵):
            coordinates: 复数坐标 (N,)
            state: 状态数组 (N,) float
            wigner: Wigner值数组 (N,) float
        """
        coordinates = self.coordinates.copy()
        state = np.array(list(self.state_matrix[1, :]), dtype=np.float64)
        wigner = np.array(list(self.state_matrix[2, :]), dtype=np.float64)
        return coordinates, state, wigner
    
    def update_from_matlab_matrix(self, state: np.ndarray, wigner: np.ndarray):
        """
        从 MATLAB 返回的矩阵更新本地状态
        
        参数:
            state: 状态数组 (N,) - MATLAB 会将状态=1的点改为2
            wigner: Wigner值数组 (N,) - MATLAB 会填充采样值
        """
        state = state.flatten()
        wigner = wigner.flatten()
        
        # 找出新采样的点 (之前是1，现在是2)
        old_state = np.array(list(self.state_matrix[1, :]), dtype=np.float64)
        new_sampled = (old_state == self.STATE_TO_SAMPLE) & (state == self.STATE_SAMPLED)
        new_indices = np.where(new_sampled)[0]
        
        # 更新状态矩阵
        for i in range(self.n_points):
            self.state_matrix[1, i] = state[i]
            self.state_matrix[2, i] = wigner[i]
        
        # 记录本轮采样
        if len(new_indices) > 0:
            self.sampling_history.append(new_indices.copy())
    
    def execute_sampling_matlab(self, state_type: int = 2, noise_std: float = 0.02,
                                  **state_params) -> bool:
        """
        使用 MATLAB 执行采样：发送完整 3×N 矩阵，MATLAB 采样后返回更新的矩阵
        
        传输协议:
            Python -> MATLAB:
                - tomo_coords: 复数坐标 (N,)
                - tomo_state: 状态数组 (N,) [0=不采, 1=请采, 2=已采]
                - tomo_wigner: Wigner值 (N,)
            
            MATLAB 处理:
                - 对 state==1 的点计算 Wigner 值
                - 将计算结果填入 wigner 数组
                - 将 state 从 1 改为 2
            
            MATLAB -> Python:
                - tomo_state: 更新后的状态数组
                - tomo_wigner: 更新后的 Wigner 值数组
        
        参数:
            state_type: 态类型 (1=Fock, 2=相干态, 3=猫态)
            noise_std: 噪声标准差
            **state_params: 态参数 (alpha, n 等)
        
        返回:
            是否成功
        """
        if self.matlab_bridge is None or not self.matlab_bridge.is_connected:
            print("⚠ MATLAB 未连接，无法采样")
            return False
        
        # 获取完整状态矩阵
        coords, state, wigner = self.get_full_state_matrix_for_matlab()
        
        # 检查是否有需要采样的点
        n_to_sample = int(np.sum(state == self.STATE_TO_SAMPLE))
        if n_to_sample == 0:
            return True
        
        print(f"  [发送] 完整矩阵 3×{self.n_points} 到 MATLAB...")
        
        # 发送完整矩阵给 MATLAB (3 行)
        self.matlab_bridge.send('tomo_coords', coords.reshape(1, -1))  # (1, N) 复数
        self.matlab_bridge.send('tomo_state', state.reshape(1, -1))    # (1, N) 实数
        self.matlab_bridge.send('tomo_wigner', wigner.reshape(1, -1))  # (1, N) 实数
        
        # 根据态类型构建 MATLAB 采样命令
        alpha = state_params.get('alpha', 2.0)
        n = state_params.get('n', 3)
        
        # MATLAB 代码: 处理完整矩阵
        if state_type == 2:  # 相干态
            wigner_formula = f'''
                sqrt2 = sqrt(2);
                x0 = 0;
                p0 = sqrt2 * {alpha};
                X = real(tomo_coords(to_sample));
                P = imag(tomo_coords(to_sample));
                W = (1/pi) * exp(-((X - x0).^2 + (P - p0).^2));
                W = W + {noise_std} * randn(size(W));
            '''
        elif state_type == 1:  # Fock 态
            wigner_formula = f'''
                X = real(tomo_coords(to_sample));
                P = imag(tomo_coords(to_sample));
                r2 = X.^2 + P.^2;
                L_n = laguerreL({n}, 0, 2*r2);
                W = ((-1)^{n} / pi) * L_n .* exp(-r2);
                W = W + {noise_std} * randn(size(W));
            '''
        elif state_type == 3:  # 猫态
            wigner_formula = f'''
                sqrt2 = sqrt(2);
                X = real(tomo_coords(to_sample));
                P = imag(tomo_coords(to_sample));
                W_plus = (1/pi) * exp(-((X - sqrt2*{alpha}).^2 + P.^2));
                W_minus = (1/pi) * exp(-((X + sqrt2*{alpha}).^2 + P.^2));
                interference = (2/pi) * exp(-(X.^2 + P.^2)) .* cos(2*sqrt2*{alpha}*P);
                N2_even = 2 * (1 + exp(-2*{alpha}^2));
                W = (W_plus + W_minus + interference * exp(-{alpha}^2)) / N2_even;
                W = W + {noise_std} * randn(size(W));
            '''
        else:
            print(f"⚠ 不支持的态类型: {state_type}")
            return False
        
        matlab_code = f'''
            % 找到需要采样的点 (state == 1)
            to_sample = (tomo_state == 1);
            n_to_sample = sum(to_sample);
            
            if n_to_sample > 0
                % 计算 Wigner 函数值
                {wigner_formula}
                
                % 填入结果
                tomo_wigner(to_sample) = W;
                
                % 更新状态: 1 -> 2
                tomo_state(to_sample) = 2;
            end
            
            disp(['已采样 ' num2str(n_to_sample) ' 个点']);
        '''
        
        # 执行 MATLAB 代码
        self.matlab_bridge.eval(matlab_code)
        
        # 接收更新后的矩阵
        print(f"  [接收] 更新后的矩阵从 MATLAB...")
        new_state = self.matlab_bridge.receive('tomo_state')
        new_wigner = self.matlab_bridge.receive('tomo_wigner')
        
        if new_state is None or new_wigner is None:
            print("⚠ MATLAB 返回数据失败")
            return False
        
        # 更新本地状态
        self.update_from_matlab_matrix(new_state, new_wigner)
        
        print(f"  ✓ 采样完成，新增 {n_to_sample} 点")
        return True
