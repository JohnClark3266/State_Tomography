"""
MATLAB数据交换接口

支持与MATLAB用户协作:
- 导出采样请求矩阵 (告诉合作者在哪采样)
- 导入实验数据矩阵 (接收合作者的测量结果)
- 继续优化循环

矩阵格式 (3, N) 其中 N = grid_size * grid_size:
- Row 0: 复数坐标 alpha = Re + i*Im (MATLAB中存为复数)
- Row 1: 采样状态 (0=未采样/不采, 1=未采样/请采, 2=已采样)
- Row 2: Wigner值 (状态=2时有效)

工作流程:
1. Python端: 确定下一轮采样点 -> export_sampling_request()
2. MATLAB端: 读取.mat文件，在状态=1的点进行测量，填入Wigner值，状态改为2
3. Python端: import_experimental_data() -> 继续训练
"""

import numpy as np
import scipy.io as sio
import os
from typing import Tuple, Optional
from datetime import datetime


class MatlabInterface:
    """MATLAB数据交换接口"""
    
    def __init__(self, grid_size: int = 64, save_dir: str = "matlab_exchange"):
        """
        初始化接口
        
        参数:
            grid_size: 网格大小
            save_dir: 数据交换目录
        """
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def export_sampling_request(self, 
                                 sampling_manager, 
                                 filename: str = None,
                                 round_id: int = None) -> str:
        """
        导出采样请求给MATLAB
        
        参数:
            sampling_manager: SamplingManager实例
            filename: 输出文件名 (不含扩展名)
            round_id: 轮次ID (用于文件名)
        
        返回:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if round_id is not None:
                filename = f"sampling_request_round{round_id}_{timestamp}"
            else:
                filename = f"sampling_request_{timestamp}"
        
        # 获取数据
        coordinates = sampling_manager.coordinates  # complex array
        state = sampling_manager.get_state()  # float array
        wigner = sampling_manager.get_wigner_values()  # float array
        
        # 构建MATLAB格式的数据字典
        matlab_data = {
            'coordinates': coordinates.reshape(1, -1),  # (1, N) complex
            'state': state.reshape(1, -1),              # (1, N) float
            'wigner_values': wigner.reshape(1, -1),     # (1, N) float
            'grid_size': np.array([self.grid_size]),
            'n_points': np.array([self.n_points]),
            # 额外信息
            'state_description': np.array(['0=unsampled_skip, 1=unsampled_please_sample, 2=sampled']),
            'instructions': np.array([
                'For points where state==1, please measure Wigner value and fill in wigner_values.',
                'After measurement, change state from 1 to 2.',
                'Save as new .mat file and send back.'
            ]),
            # 坐标网格 (方便MATLAB可视化)
            'X_grid': sampling_manager.X_flat.reshape(self.grid_size, self.grid_size),
            'P_grid': sampling_manager.P_flat.reshape(self.grid_size, self.grid_size),
        }
        
        # 统计信息
        n_to_sample = int(np.sum(state == 1))
        n_sampled = int(np.sum(state == 2))
        matlab_data['n_to_sample'] = np.array([n_to_sample])
        matlab_data['n_sampled'] = np.array([n_sampled])
        
        filepath = os.path.join(self.save_dir, f"{filename}.mat")
        sio.savemat(filepath, matlab_data)
        
        print(f"\n{'='*60}")
        print(f"采样请求已导出: {filepath}")
        print(f"{'='*60}")
        print(f"  待采样点数: {n_to_sample}")
        print(f"  已采样点数: {n_sampled}")
        print(f"  总点数: {self.n_points}")
        print(f"\n请将此文件发送给MATLAB用户。")
        print(f"合作者需要:")
        print(f"  1. 读取.mat文件")
        print(f"  2. 在state==1的点进行测量")
        print(f"  3. 将测量值填入wigner_values对应位置")
        print(f"  4. 将state从1改为2")
        print(f"  5. 保存新的.mat文件并返回")
        print(f"{'='*60}\n")
        
        return filepath
    
    def import_experimental_data(self, 
                                  filepath: str, 
                                  sampling_manager) -> Tuple[int, int]:
        """
        从MATLAB导入实验数据
        
        参数:
            filepath: .mat文件路径
            sampling_manager: SamplingManager实例 (将被更新)
        
        返回:
            (新增采样点数, 总采样点数)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 读取MATLAB数据
        matlab_data = sio.loadmat(filepath)
        
        # 提取数据
        new_state = matlab_data['state'].flatten()
        new_wigner = matlab_data['wigner_values'].flatten()
        
        # 验证尺寸
        if len(new_state) != self.n_points:
            raise ValueError(f"状态数组长度错误: {len(new_state)} != {self.n_points}")
        
        # 获取旧状态
        old_state = sampling_manager.get_state()
        old_n_sampled = int(np.sum(old_state == 2))
        
        # 找出新采样的点 (状态从1变成2的点)
        new_sampled_mask = (old_state == 1) & (new_state == 2)
        n_new_samples = int(np.sum(new_sampled_mask))
        
        # 更新采样管理器
        for i in range(self.n_points):
            if new_sampled_mask[i]:
                sampling_manager.state_matrix[1, i] = 2  # 状态改为已采样
                sampling_manager.state_matrix[2, i] = new_wigner[i]  # 填入Wigner值
        
        new_n_sampled = int(np.sum(np.array(list(sampling_manager.state_matrix[1, :]), dtype=float) == 2))
        
        print(f"\n{'='*60}")
        print(f"实验数据已导入: {filepath}")
        print(f"{'='*60}")
        print(f"  新增采样点: {n_new_samples}")
        print(f"  原有采样点: {old_n_sampled}")
        print(f"  当前总采样点: {new_n_sampled}")
        print(f"  采样率: {new_n_sampled/self.n_points*100:.2f}%")
        print(f"{'='*60}\n")
        
        return n_new_samples, new_n_sampled
    
    def create_initial_matrix(self, 
                               x_range: Tuple[float, float] = (-5, 5)
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建初始矩阵 (用于首次协作)
        
        返回:
            coordinates: 复数坐标数组 (N,)
            state: 状态数组 (N,)
            wigner: Wigner值数组 (N,)
        """
        x = np.linspace(x_range[0], x_range[1], self.grid_size)
        p = np.linspace(x_range[0], x_range[1], self.grid_size)
        X, P = np.meshgrid(x, p)
        
        coordinates = (X + 1j * P).flatten()
        state = np.zeros(self.n_points)
        wigner = np.zeros(self.n_points)
        
        return coordinates, state, wigner


def export_for_matlab(sampling_manager, round_id: int = None, save_dir: str = "matlab_exchange") -> str:
    """
    便捷函数: 导出采样请求给MATLAB
    
    用法:
        from matlab_interface import export_for_matlab
        filepath = export_for_matlab(tomo.sampler, round_id=1)
    """
    interface = MatlabInterface(sampling_manager.grid_size, save_dir)
    return interface.export_sampling_request(sampling_manager, round_id=round_id)


def import_from_matlab(filepath: str, sampling_manager, save_dir: str = "matlab_exchange") -> Tuple[int, int]:
    """
    便捷函数: 从MATLAB导入实验数据
    
    用法:
        from matlab_interface import import_from_matlab
        n_new, n_total = import_from_matlab("data.mat", tomo.sampler)
    """
    interface = MatlabInterface(sampling_manager.grid_size, save_dir)
    return interface.import_experimental_data(filepath, sampling_manager)


# ============== MATLAB端示例代码 ==============
MATLAB_EXAMPLE_CODE = """
%% MATLAB端示例代码

% 1. 读取Python发送的采样请求
data = load('sampling_request_round1.mat');

% 2. 查看需要采样的点
to_sample = find(data.state == 1);
fprintf('需要采样 %d 个点\\n', length(to_sample));

% 3. 获取需要采样的坐标
coords_to_sample = data.coordinates(to_sample);
re_alpha = real(coords_to_sample);  % x坐标
im_alpha = imag(coords_to_sample);  % p坐标

% 4. 进行实验测量 (这里用您的实验代码)
% measured_wigner = your_experiment_function(re_alpha, im_alpha);

% 5. 模拟测量 (实际使用时替换为真实数据)
measured_wigner = randn(size(to_sample)) * 0.1;  % 示例

% 6. 填入测量结果
data.wigner_values(to_sample) = measured_wigner;
data.state(to_sample) = 2;  % 状态改为已采样

% 7. 保存并发回给Python
save('experimental_data_round1.mat', '-struct', 'data');
fprintf('数据已保存，请发送给Python端\\n');
"""


if __name__ == "__main__":
    # 测试代码
    print("MATLAB接口模块")
    print("=" * 60)
    print("\nMATLAB端示例代码:")
    print(MATLAB_EXAMPLE_CODE)
