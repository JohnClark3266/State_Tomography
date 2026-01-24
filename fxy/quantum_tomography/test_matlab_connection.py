"""测试 MATLAB Engine 连接"""
from matlab_bridge import MatlabBridge
import numpy as np

print("正在连接 MATLAB...")
bridge = MatlabBridge()

if bridge.is_connected:
    # 测试发送数据
    test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    bridge.send('test_matrix', test_data)
    
    # 让 MATLAB 计算这个矩阵的特征值
    result = bridge.call('eig', test_data, nargout=1)
    print(f"特征值: {result.flatten()}")
    
    # 在 MATLAB 中执行命令
    bridge.eval('disp("Hello from Python!")')
    
    print("\n✓ 所有测试通过!")
    print("\n现在可以在 MATLAB 中查看 test_matrix 变量")
else:
    print("✗ 无法连接到 MATLAB")
