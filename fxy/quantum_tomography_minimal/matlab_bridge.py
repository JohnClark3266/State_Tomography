"""
MATLAB Engine Bridge - 自动矩阵同步

使用 MATLAB Engine for Python 实现 Python <-> MATLAB 的自动变量传输。
不需要手动保存/加载 .mat 文件，变量直接在内存中共享。

安装要求:
    1. 安装 MATLAB (需要有效许可证)
    2. 在 MATLAB 中运行:
       cd (fullfile(matlabroot,'extern','engines','python'))
       system('python -m pip install .')
    
    或者用命令行:
       cd /Applications/MATLAB_R2024a.app/extern/engines/python
       pip install .

用法:
    from matlab_bridge import MatlabBridge
    
    # 连接 MATLAB
    bridge = MatlabBridge()
    
    # Python -> MATLAB
    bridge.send('my_matrix', numpy_array)
    
    # MATLAB -> Python
    data = bridge.receive('matlab_variable')
    
    # 自动同步 (双向)
    bridge.sync('shared_data', numpy_array)
    result = bridge.sync('shared_data')  # 获取 MATLAB 端修改后的值
"""

import numpy as np
from typing import Optional, Union, Dict, Any
import warnings


class MatlabBridge:
    """MATLAB Engine 桥接器 - 自动矩阵同步"""
    
    def __init__(self, connect_existing: bool = True, session_name: str = None):
        """
        初始化 MATLAB 连接
        
        参数:
            connect_existing: 是否连接到已运行的 MATLAB 会话
            session_name: 指定会话名 (如果 connect_existing=True)
        """
        self.engine = None
        self._connected = False
        self._shared_vars: Dict[str, np.ndarray] = {}
        
        try:
            import matlab.engine
            self._matlab_module = matlab.engine
            self._connect(connect_existing, session_name)
        except ImportError:
            warnings.warn(
                "未安装 MATLAB Engine for Python。\n"
                "安装方法:\n"
                "  cd /Applications/MATLAB_R2024a.app/extern/engines/python\n"
                "  pip install .\n"
                "将启用离线模式 (仅文件交换)。"
            )
            self._matlab_module = None
    
    def _connect(self, connect_existing: bool, session_name: str = None):
        """连接到 MATLAB"""
        if self._matlab_module is None:
            return
        
        try:
            if connect_existing:
                # 尝试连接到已有会话
                sessions = self._matlab_module.find_matlab()
                if sessions:
                    if session_name and session_name in sessions:
                        self.engine = self._matlab_module.connect_matlab(session_name)
                    else:
                        self.engine = self._matlab_module.connect_matlab(sessions[0])
                    print(f"✓ 已连接到 MATLAB 会话: {sessions[0] if not session_name else session_name}")
                else:
                    print("未找到运行中的 MATLAB 会话，正在启动新会话...")
                    self.engine = self._matlab_module.start_matlab()
                    print("✓ MATLAB 已启动")
            else:
                self.engine = self._matlab_module.start_matlab()
                print("✓ MATLAB 已启动")
            
            self._connected = True
            
        except Exception as e:
            warnings.warn(f"连接 MATLAB 失败: {e}")
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected and self.engine is not None
    
    def send(self, name: str, data: np.ndarray, workspace: str = 'base') -> bool:
        """
        发送矩阵到 MATLAB
        
        参数:
            name: MATLAB 中的变量名
            data: NumPy 数组
            workspace: 工作空间 ('base' 或 'global')
        
        返回:
            是否成功
        
        用法:
            bridge.send('wigner_data', my_numpy_array)
            # 在 MATLAB 中: wigner_data 已经可用
        """
        if not self.is_connected:
            print("✗ 未连接到 MATLAB")
            return False
        
        try:
            import matlab
            
            # 转换为 MATLAB 格式
            if np.iscomplexobj(data):
                # 复数数组
                matlab_data = matlab.double(data.tolist(), is_complex=True)
            else:
                # 实数数组
                matlab_data = matlab.double(data.tolist())
            
            # 发送到 MATLAB 工作空间
            self.engine.workspace[name] = matlab_data
            
            # 记录到共享变量
            self._shared_vars[name] = data.copy()
            
            print(f"✓ 已发送 '{name}' 到 MATLAB ({data.shape})")
            return True
            
        except Exception as e:
            print(f"✗ 发送失败: {e}")
            return False
    
    def receive(self, name: str, workspace: str = 'base') -> Optional[np.ndarray]:
        """
        从 MATLAB 接收矩阵
        
        参数:
            name: MATLAB 中的变量名
            workspace: 工作空间
        
        返回:
            NumPy 数组，失败返回 None
        
        用法:
            # 在 MATLAB 中: result = some_calculation(wigner_data);
            result = bridge.receive('result')
        """
        if not self.is_connected:
            print("✗ 未连接到 MATLAB")
            return None
        
        try:
            # 从 MATLAB 工作空间获取
            matlab_data = self.engine.workspace[name]
            
            # 转换为 NumPy
            data = np.array(matlab_data)
            
            # 更新共享变量
            self._shared_vars[name] = data.copy()
            
            print(f"✓ 已接收 '{name}' 从 MATLAB ({data.shape})")
            return data
            
        except Exception as e:
            print(f"✗ 接收失败: {e}")
            return None
    
    def sync(self, name: str, data: np.ndarray = None) -> Optional[np.ndarray]:
        """
        双向同步变量
        
        如果提供 data，则发送到 MATLAB；
        否则从 MATLAB 接收最新值。
        
        参数:
            name: 变量名
            data: 要发送的数据 (可选)
        
        返回:
            当前值 (来自 MATLAB 或本地)
        
        用法:
            # 发送数据
            bridge.sync('shared_matrix', my_data)
            
            # ... MATLAB 端进行修改 ...
            
            # 获取修改后的数据
            updated = bridge.sync('shared_matrix')
        """
        if data is not None:
            self.send(name, data)
            return data
        else:
            return self.receive(name)
    
    def call(self, func_name: str, *args, nargout: int = 1) -> Any:
        """
        调用 MATLAB 函数
        
        参数:
            func_name: 函数名
            *args: 参数 (自动转换 NumPy 数组)
            nargout: 输出参数个数
        
        返回:
            函数返回值
        
        用法:
            # 调用 MATLAB 的 eig 函数
            eigenvalues = bridge.call('eig', my_matrix, nargout=1)
            
            # 调用自定义函数
            result = bridge.call('my_wigner_function', alpha, nargout=1)
        """
        if not self.is_connected:
            print("✗ 未连接到 MATLAB")
            return None
        
        try:
            import matlab
            
            # 转换参数
            converted_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if np.iscomplexobj(arg):
                        converted_args.append(matlab.double(arg.tolist(), is_complex=True))
                    else:
                        converted_args.append(matlab.double(arg.tolist()))
                else:
                    converted_args.append(arg)
            
            # 调用函数
            func = getattr(self.engine, func_name)
            result = func(*converted_args, nargout=nargout)
            
            # 转换结果
            if nargout == 1:
                return np.array(result) if hasattr(result, '__iter__') else result
            else:
                return tuple(np.array(r) if hasattr(r, '__iter__') else r for r in result)
                
        except Exception as e:
            print(f"✗ 调用 '{func_name}' 失败: {e}")
            return None
    
    def eval(self, command: str) -> None:
        """
        在 MATLAB 中执行命令
        
        参数:
            command: MATLAB 命令字符串
        
        用法:
            bridge.eval("figure; imagesc(wigner_data); colorbar;")
        """
        if not self.is_connected:
            print("✗ 未连接到 MATLAB")
            return
        
        try:
            self.engine.eval(command, nargout=0)
            print(f"✓ 已执行: {command[:50]}...")
        except Exception as e:
            print(f"✗ 执行失败: {e}")
    
    def list_shared(self) -> Dict[str, tuple]:
        """列出所有共享变量"""
        return {name: arr.shape for name, arr in self._shared_vars.items()}
    
    def close(self):
        """关闭 MATLAB 连接"""
        if self.engine is not None:
            try:
                self.engine.quit()
                print("✓ MATLAB 连接已关闭")
            except:
                pass
        self._connected = False
        self.engine = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __del__(self):
        self.close()


class WignerBridge(MatlabBridge):
    """
    专用于 Wigner 函数数据交换的桥接器
    
    提供更方便的接口用于量子态层析
    """
    
    def __init__(self, grid_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
    
    def send_sampling_request(self, coordinates: np.ndarray, 
                               state: np.ndarray,
                               wigner_values: np.ndarray,
                               var_prefix: str = 'tomo') -> bool:
        """
        发送采样请求到 MATLAB
        
        参数:
            coordinates: 复数坐标 (N,)
            state: 状态数组 (N,)
            wigner_values: Wigner值 (N,)
            var_prefix: 变量名前缀
        """
        success = True
        success &= self.send(f'{var_prefix}_coords', coordinates)
        success &= self.send(f'{var_prefix}_state', state)
        success &= self.send(f'{var_prefix}_wigner', wigner_values)
        
        if success:
            # 在 MATLAB 中创建便捷的结构体
            self.eval(f"""
                {var_prefix} = struct();
                {var_prefix}.coordinates = {var_prefix}_coords;
                {var_prefix}.state = {var_prefix}_state;
                {var_prefix}.wigner_values = {var_prefix}_wigner;
                {var_prefix}.grid_size = {self.grid_size};
                {var_prefix}.to_sample = find({var_prefix}_state == 1);
                disp(['待采样点数: ' num2str(length({var_prefix}.to_sample))]);
            """)
        
        return success
    
    def receive_experimental_data(self, var_prefix: str = 'tomo') -> Optional[tuple]:
        """
        从 MATLAB 接收实验数据
        
        返回:
            (state, wigner_values) 或 None
        """
        state = self.receive(f'{var_prefix}_state')
        wigner = self.receive(f'{var_prefix}_wigner')
        
        if state is not None and wigner is not None:
            return state.flatten(), wigner.flatten()
        return None
    
    def auto_sync_loop(self, sampling_manager, callback=None, poll_interval: float = 1.0):
        """
        自动同步循环
        
        持续监控 MATLAB 端的更新，自动同步数据
        
        参数:
            sampling_manager: SamplingManager 实例
            callback: 数据更新时的回调函数
            poll_interval: 轮询间隔 (秒)
        """
        import time
        
        print(f"开始自动同步... (每 {poll_interval}s 检查一次)")
        print("按 Ctrl+C 停止")
        
        last_n_sampled = 0
        
        try:
            while True:
                # 检查 MATLAB 端的更新
                result = self.receive_experimental_data()
                if result is not None:
                    state, wigner = result
                    n_sampled = int(np.sum(state == 2))
                    
                    if n_sampled > last_n_sampled:
                        print(f"\n检测到新数据! 采样点: {last_n_sampled} -> {n_sampled}")
                        # 更新 sampling_manager
                        for i in range(len(state)):
                            if state[i] == 2:
                                sampling_manager.state_matrix[1, i] = 2
                                sampling_manager.state_matrix[2, i] = wigner[i]
                        
                        last_n_sampled = n_sampled
                        
                        if callback:
                            callback(sampling_manager)
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\n同步已停止")


# ============== 便捷函数 ==============

def quick_connect() -> MatlabBridge:
    """快速连接到 MATLAB"""
    return MatlabBridge(connect_existing=True)


def send_to_matlab(name: str, data: np.ndarray) -> bool:
    """一次性发送数据到 MATLAB"""
    with MatlabBridge() as bridge:
        return bridge.send(name, data)


def get_from_matlab(name: str) -> Optional[np.ndarray]:
    """一次性从 MATLAB 获取数据"""
    with MatlabBridge() as bridge:
        return bridge.receive(name)


# ============== 使用示例 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("MATLAB Bridge - 自动矩阵同步")
    print("=" * 60)
    
    # 创建示例数据
    test_matrix = np.random.randn(64, 64)
    complex_coords = np.linspace(-5, 5, 100) + 1j * np.linspace(-5, 5, 100)
    
    print("\n示例用法:")
    print("-" * 60)
    print("""
    # 1. 基本用法
    from matlab_bridge import MatlabBridge
    
    bridge = MatlabBridge()
    
    # 发送数据到 MATLAB
    bridge.send('wigner_data', my_numpy_array)
    
    # 从 MATLAB 接收数据
    result = bridge.receive('matlab_result')
    
    # 调用 MATLAB 函数
    eigenvalues = bridge.call('eig', my_matrix)
    
    # 执行 MATLAB 命令
    bridge.eval("figure; imagesc(wigner_data);")
    
    # 2. 与层析代码集成
    from matlab_bridge import WignerBridge
    
    bridge = WignerBridge(grid_size=64)
    bridge.send_sampling_request(coords, state, wigner)
    
    # ... MATLAB 端测量 ...
    
    state, wigner = bridge.receive_experimental_data()
    """)
    
    # 尝试连接
    print("\n尝试连接 MATLAB...")
    bridge = MatlabBridge()
    
    if bridge.is_connected:
        print("\n测试发送数据...")
        bridge.send('test_matrix', test_matrix)
        bridge.send('test_coords', complex_coords)
        
        print("\n在 MATLAB 中可以访问:")
        print("  - test_matrix (64x64 实数矩阵)")
        print("  - test_coords (100x1 复数数组)")
        
        bridge.close()
    else:
        print("\n无法连接到 MATLAB，请确保:")
        print("  1. MATLAB 已安装并启动")
        print("  2. 已安装 MATLAB Engine for Python")
        print("  3. 在 MATLAB 中运行: matlab.engine.shareEngine")
