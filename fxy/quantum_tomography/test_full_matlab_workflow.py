"""
完整 MATLAB 采样工作流测试

这个脚本演示如何:
1. Python 决定采样点
2. 发送给 MATLAB 执行采样 (使用 MATLAB 的 Wigner 函数计算)
3. MATLAB 返回测量结果
4. Python 继续训练和迭代

模式选择:
- use_matlab=True: 真正调用 MATLAB 采样
- use_matlab=False: 用 Python 模拟 (调试用)
"""

import numpy as np
import torch
import time
from sampling_manager import SamplingManager
from decision_maker import DecisionMaker
from quantum_states import create_state
from neural_networks import build_model_pool, select_committee
from noise_model import compute_fidelity
from matlab_bridge import MatlabBridge

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MatlabIntegratedTomography:
    """MATLAB 集成的主动学习层析"""
    
    def __init__(self, grid_size=64, state_type=2, use_matlab=True,
                 samples_per_round=20, max_rounds=10, F_threshold=0.98,
                 pretrain_epochs=30, finetune_epochs=10, **state_params):
        
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.use_matlab = use_matlab
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.F_threshold = F_threshold
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        
        # 设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 生成目标态 (理论值，用于验证)
        self.X, self.P, self.ideal_wigner, self.state_name = create_state(
            grid_size, state_type, **state_params
        )
        print(f"目标态: {self.state_name}")
        
        # 初始化管理器
        self.sampler = SamplingManager(grid_size)
        self.decision = DecisionMaker(grid_size)
        
        # 构建委员会
        print("\n构建神经网络委员会...")
        self.model_pool = build_model_pool(grid_size)
        self.committee = select_committee(self.model_pool, 5)
        print(f"委员会成员: {[name for name, _ in self.committee]}")
        
        for name, model in self.committee:
            model.to(self.device)
        
        # MATLAB 连接
        self.bridge = None
        if use_matlab:
            print("\n连接 MATLAB...")
            self.bridge = MatlabBridge()
            if self.bridge.is_connected:
                self._setup_matlab()
            else:
                print("⚠ MATLAB 连接失败，切换到模拟模式")
                self.use_matlab = False
        
        self.history = []
    
    def _setup_matlab(self):
        """在 MATLAB 中设置 Wigner 函数采样代码"""
        # 发送网格信息
        self.bridge.send('grid_size', np.array([[self.grid_size]]))
        self.bridge.send('X_grid', self.X)
        self.bridge.send('P_grid', self.P)
        
        # 定义 MATLAB 端的采样函数 (基于坐标)
        matlab_code = '''
        % 创建 Wigner 采样函数 (用于演示)
        % 这里用高斯模拟，实际使用时替换为真实实验采样
        
        function w = sample_wigner_at_points(X, P, alpha, state_type)
            % state_type: 2 = 相干态
            if state_type == 2
                % 相干态 Wigner 函数
                x0 = real(alpha);
                p0 = imag(alpha);
                w = (2/pi) * exp(-2*((X - x0).^2 + (P - p0).^2));
            else
                w = zeros(size(X));
            end
            % 添加实验噪声
            w = w + 0.02 * randn(size(w));
        end
        '''
        # 注意: 这里只是演示，实际应该用真实的 MATLAB 函数
        print("✓ MATLAB 采样环境已设置")
    
    def _matlab_sample(self, indices):
        """使用 MATLAB 采样指定点"""
        if not self.use_matlab or not self.bridge.is_connected:
            # 回退到 Python 模拟
            return self._python_sample(indices)
        
        # 获取要采样的坐标
        coords = self.sampler.coordinates[indices]
        X_sample = np.real(coords)
        P_sample = np.imag(coords)
        
        # 发送给 MATLAB
        self.bridge.send('sample_X', X_sample.reshape(-1, 1))
        self.bridge.send('sample_P', P_sample.reshape(-1, 1))
        
        # 在 MATLAB 中计算 Wigner 值 (模拟相干态)
        alpha = self.state_params.get('alpha', 2.0)
        self.bridge.eval(f'''
            alpha = {alpha};
            x0 = real(alpha);
            p0 = imag(alpha);
            % 相干态 Wigner 函数
            wigner_values = (2/pi) * exp(-2*((sample_X - x0).^2 + (sample_P - p0).^2));
            % 添加实验噪声
            wigner_values = wigner_values + 0.02 * randn(size(wigner_values));
        ''')
        
        # 从 MATLAB 获取结果
        wigner_values = self.bridge.receive('wigner_values')
        
        if wigner_values is not None:
            return wigner_values.flatten()
        else:
            print("⚠ MATLAB 采样失败，使用 Python 备份")
            return self._python_sample(indices)
    
    def _python_sample(self, indices):
        """Python 模拟采样 (调试用)"""
        values = self.ideal_wigner.flatten()[indices]
        # 添加噪声模拟实验
        values = values + np.random.normal(0, 0.02, len(values))
        return values
    
    def _pretrain(self):
        """预训练委员会"""
        print(f"\n[预训练] {self.pretrain_epochs} epochs...")
        
        # 生成训练数据
        n_samples = 100
        inputs = []
        targets = []
        
        for _ in range(n_samples):
            ratio = np.random.uniform(0.02, 0.30)
            n_sample = int(self.grid_size ** 2 * ratio)
            indices = np.random.choice(self.grid_size ** 2, n_sample, replace=False)
            
            sparse = np.zeros((self.grid_size, self.grid_size))
            mask = np.zeros((self.grid_size, self.grid_size))
            
            flat_sparse = sparse.flatten()
            flat_mask = mask.flatten()
            flat_sparse[indices] = self.ideal_wigner.flatten()[indices]
            flat_mask[indices] = 1.0
            
            sparse = flat_sparse.reshape(self.grid_size, self.grid_size)
            mask = flat_mask.reshape(self.grid_size, self.grid_size)
            
            inp = np.stack([sparse, mask], axis=0).astype(np.float32)
            inputs.append(inp)
            targets.append(self.ideal_wigner[np.newaxis, :, :].astype(np.float32))
        
        train_inputs = torch.tensor(np.array(inputs))
        train_targets = torch.tensor(np.array(targets))
        dataset = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(self.pretrain_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
        
        print("✓ 预训练完成")
    
    def _committee_predict(self, sparse_input):
        """委员会预测"""
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :], 
                                     dtype=torch.float32).to(self.device)
        
        predictions = []
        for name, model in self.committee:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.var(axis=0)
    
    def _finetune(self):
        """微调"""
        sparse_input = self.sampler.get_sparse_input_for_nn()
        target = self.ideal_wigner[np.newaxis, :, :].astype(np.float32)
        
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :]).to(self.device)
        target_tensor = torch.tensor(target[np.newaxis, :, :, :]).to(self.device)
        
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            for _ in range(self.finetune_epochs):
                optimizer.zero_grad()
                pred = model(input_tensor)
                loss = criterion(pred, target_tensor)
                loss.backward()
                optimizer.step()
    
    def run(self):
        """运行完整工作流"""
        print("\n" + "="*60)
        print("MATLAB 集成层析测试")
        print("="*60)
        print(f"MATLAB 模式: {'启用' if self.use_matlab else '模拟'}")
        print(f"目标态: {self.state_name}")
        print(f"每轮采样: {self.samples_per_round} 点")
        print("="*60 + "\n")
        
        # 1. 预训练
        self._pretrain()
        
        # 2. 初始采样
        print("\n[初始化] 选择初始采样点...")
        empty_input = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        mean_pred, variance = self._committee_predict(empty_input)
        
        n_initial = int(self.grid_size ** 2 * 0.02)
        initial_indices = self.decision.decide_initial_samples(
            variance.flatten(), mean_pred.flatten(), n_initial
        )
        
        # 使用 MATLAB 采样
        print(f"  采样 {len(initial_indices)} 个初始点...")
        wigner_values = self._matlab_sample(initial_indices)
        
        # 更新 sampler
        for i, idx in enumerate(initial_indices):
            self.sampler.state_matrix[1, idx] = 2  # 已采样
            self.sampler.state_matrix[2, idx] = wigner_values[i]
        
        print(f"  ✓ 初始采样完成")
        
        # 3. 主循环
        for round_id in range(self.max_rounds):
            n_sampled = self.sampler.get_sampled_count()
            ratio = self.sampler.get_sampling_ratio()
            print(f"\n[Round {round_id+1}/{self.max_rounds}] 采样点: {n_sampled} ({ratio*100:.1f}%)")
            
            # 微调
            self._finetune()
            
            # 预测
            sparse_input = self.sampler.get_sparse_input_for_nn()
            mean_pred, variance = self._committee_predict(sparse_input)
            
            # 用测量值替换
            mask = self.sampler.get_mask_2d()
            measured = self.sampler.get_wigner_2d()
            mean_pred[mask] = measured[mask]
            
            # 保真度
            F_ideal = compute_fidelity(mean_pred, self.ideal_wigner)
            
            self.history.append({
                'round': round_id + 1,
                'n_sampled': n_sampled,
                'ratio': ratio,
                'fidelity': F_ideal
            })
            
            print(f"  保真度 (vs理论): {F_ideal:.5f}")
            
            if F_ideal >= self.F_threshold:
                print(f"\n✓ 达到目标保真度 {self.F_threshold}!")
                break
            
            # 下一轮采样
            if round_id < self.max_rounds - 1:
                current_state = self.sampler.get_state()
                next_indices = self.decision.decide_next_samples(
                    variance.flatten(), mean_pred.flatten(),
                    current_state, self.samples_per_round
                )
                
                print(f"  采样 {len(next_indices)} 个新点...")
                wigner_values = self._matlab_sample(next_indices)
                
                for i, idx in enumerate(next_indices):
                    self.sampler.state_matrix[1, idx] = 2
                    self.sampler.state_matrix[2, idx] = wigner_values[i]
        
        # 4. 最终结果
        print("\n" + "="*60)
        print("完成!")
        print(f"最终采样率: {self.sampler.get_sampling_ratio()*100:.1f}%")
        print(f"最终保真度: {self.history[-1]['fidelity']:.5f}")
        print("="*60)
        
        # 关闭 MATLAB 连接
        if self.bridge:
            self.bridge.close()
        
        return self.history


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试 (use_matlab=True 使用 MATLAB 采样)
    tomo = MatlabIntegratedTomography(
        grid_size=64,
        state_type=2,  # 相干态
        alpha=2.0,
        use_matlab=True,  # 启用 MATLAB
        samples_per_round=30,
        max_rounds=15,
        F_threshold=0.98,
        pretrain_epochs=30,
        finetune_epochs=10
    )
    
    history = tomo.run()
    
    print("\n保真度历史:")
    for h in history:
        print(f"  Round {h['round']:2d}: ratio={h['ratio']*100:5.1f}%, F={h['fidelity']:.5f}")
