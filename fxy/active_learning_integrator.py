import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 神经网络模型定义 ====================

class SineActivation(nn.Module):
    """正弦激活函数（用于SirenNet）"""
    def __init__(self, w0=30.0):
        super(SineActivation, self).__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)


class MLP(nn.Module):
    """基础多层感知机"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class FourierNet(nn.Module):
    """傅里叶特征网络"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_fourier=10):
        super(FourierNet, self).__init__()
        self.fourier_b = nn.Parameter(torch.randn(input_dim, num_fourier) * 2)
        
        self.net = nn.Sequential(
            nn.Linear(2 * num_fourier, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        fourier_features = torch.cat([
            torch.sin(2 * np.pi * x @ self.fourier_b),
            torch.cos(2 * np.pi * x @ self.fourier_b)
        ], dim=-1)
        return self.net(fourier_features)


class ResidualNet(nn.Module):
    """残差网络"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_blocks=3):
        super(ResidualNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for block in self.blocks:
            residual = x
            x = self.activation(block(x) + residual)
        return self.output_layer(x)


class SirenNet(nn.Module):
    """SIREN网络"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=3, w0=30.0):
        super(SirenNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=w0))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class WideNet(nn.Module):
    """宽网络"""
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1, num_layers=2):
        super(WideNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ==================== 2. 主动学习积分器类 ====================

class ActiveLearningIntegrator:
    def __init__(self, target_function, domain, Ln=10000, S0=100, St=50, m=20):
        """
        参数:
        - target_function: 目标函数 f(x, y)
        - domain: 积分域 [(x_min, x_max), (y_min, y_max)]
        - Ln: 总点数
        - S0: 初始数据集大小
        - St: 每轮主动学习添加的点数
        - m: 最大迭代轮数
        """
        self.func = target_function
        self.domain = domain
        self.Ln = Ln
        self.S0 = S0
        self.St = St
        self.m = m
        
        # 生成所有候选点
        self.all_points = self._generate_points(Ln)
        
        # 初始化五个不同的神经网络
        self.models = self._init_models()
        
        # 存储训练数据
        self.train_points = None
        self.train_values = None
        self.trained_indices = set()  # 已经训练过的点的索引
        
        # 存储历史信息
        self.history = {
            'losses': [],
            'integral_estimates': [],
            'variances': [],
            'points_added': []
        }
    
    def _init_models(self):
        """初始化五个不同的神经网络模型"""
        models = [
            MLP(input_dim=2, hidden_dim=64, output_dim=1, num_layers=4),
            FourierNet(input_dim=2, hidden_dim=64, output_dim=1, num_fourier=10),
            ResidualNet(input_dim=2, hidden_dim=64, output_dim=1, num_blocks=3),
            SirenNet(input_dim=2, hidden_dim=64, output_dim=1, num_layers=4, w0=30.0),
            WideNet(input_dim=2, hidden_dim=128, output_dim=1, num_layers=3)
        ]
        return models
    
    def _generate_points(self, num_points):
        """在积分域内生成均匀分布的随机点"""
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        
        points_x = torch.rand(num_points) * (x_max - x_min) + x_min
        points_y = torch.rand(num_points) * (y_max - y_min) + y_min
        return torch.stack([points_x, points_y], dim=1)
    
    def _compute_exact_values(self, points):
        """计算点的精确函数值"""
        values = []
        for point in points:
            x, y = point[0].item(), point[1].item()
            values.append(self.func(x, y))
        return torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    
    def initialize_dataset(self):
        """初始化数据集"""
        # 随机选择S0个点
        indices = torch.randperm(self.Ln)[:self.S0].tolist()
        initial_points = self.all_points[indices]
        initial_values = self._compute_exact_values(initial_points)
        
        self.train_points = initial_points
        self.train_values = initial_values
        self.trained_indices.update(indices)
        
        print(f"初始化数据集: {self.S0}个点")
        return indices
    
    def train_models(self, num_epochs=200, batch_size=32, learning_rate=0.001):
        """训练所有神经网络模型"""
        losses = []
        
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(self.train_points, self.train_values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for model_idx, model in enumerate(self.models):
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            model_losses = []
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_points, batch_values in dataloader:
                    optimizer.zero_grad()
                    predictions = model(batch_points)
                    loss = criterion(predictions, batch_values)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / len(dataloader)
                model_losses.append(avg_epoch_loss)
                
                # 简单早停
                if epoch > 50 and len(model_losses) > 10:
                    if np.mean(model_losses[-5:]) > np.mean(model_losses[-10:-5]):
                        break
            
            losses.append(model_losses[-1])
        
        avg_loss = np.mean(losses)
        return avg_loss
    
    def compute_predictions_variance(self):
        """计算所有模型在所有点上的预测方差"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(self.all_points)
                all_predictions.append(predictions)
        
        # 计算方差
        predictions_tensor = torch.stack(all_predictions, dim=0)  # [5, Ln, 1]
        variances = predictions_tensor.var(dim=0).squeeze()  # [Ln]
        
        return variances.numpy()
    
    def select_new_points(self, variances):
        """选择方差最大的St个新点（排除已训练的点）"""
        # 创建掩码：True表示可选，False表示已训练
        available_mask = np.ones(self.Ln, dtype=bool)
        for idx in self.trained_indices:
            available_mask[idx] = False
        
        # 只在可选点中选择
        available_variances = variances.copy()
        available_variances[~available_mask] = -np.inf  # 已训练点设负无穷，不会被选中
        
        # 选择方差最大的St个点
        if np.sum(available_mask) < self.St:
            # 如果可选点不足，选择所有可选点
            new_indices = np.where(available_mask)[0]
        else:
            new_indices = np.argsort(available_variances)[-self.St:]
        
        return new_indices
    
    def active_learning_cycle(self):
        """执行主动学习循环"""
        print("=" * 60)
        print("开始主动学习")
        print("=" * 60)
        
        # 步骤1: 初始化数据集
        print(f"\n步骤1: 初始化 {self.S0} 个点的数据集")
        self.initialize_dataset()
        
        # 初始训练
        print(f"步骤2: 初始训练")
        loss = self.train_models(num_epochs=150)
        self.history['losses'].append(loss)
        print(f"初始训练损失: {loss:.6f}")
        
        # 主循环
        for cycle in range(self.m):
            print(f"\n{'='*40}")
            print(f"主动学习循环 {cycle + 1}/{self.m}")
            print(f"{'='*40}")
            
            # 步骤2: 计算所有点的预测方差
            print("步骤2: 计算模型预测方差...")
            variances = self.compute_predictions_variance()
            self.history['variances'].append(variances)
            
            # 步骤3: 选择方差最大的点
            print("步骤3: 选择新点...")
            new_indices = self.select_new_points(variances)
            new_points = self.all_points[new_indices]
            
            # 查询精确值
            new_values = self._compute_exact_values(new_points)
            
            # 步骤4: 添加到训练集
            print(f"步骤4: 添加 {len(new_points)} 个新点到训练集")
            self.train_points = torch.cat([self.train_points, new_points], dim=0)
            self.train_values = torch.cat([self.train_values, new_values], dim=0)
            self.trained_indices.update(new_indices.tolist())
            
            print(f"当前训练集大小: {len(self.train_points)}")
            print(f"已查询比例: {len(self.train_points)/self.Ln*100:.1f}%")
            
            # 重新训练模型
            print("重新训练所有模型...")
            loss = self.train_models(num_epochs=100)
            self.history['losses'].append(loss)
            print(f"训练损失: {loss:.6f}")
            
            # 计算积分估计
            integral_estimate = self.estimate_integral()
            self.history['integral_estimates'].append(integral_estimate)
            print(f"积分估计: {integral_estimate:.6f}")
            
            # 检查收敛
            if len(self.history['integral_estimates']) >= 3:
                recent = self.history['integral_estimates'][-3:]
                diff = max(recent) - min(recent)
                if diff < 0.01 * abs(np.mean(recent)):
                    print(f"✓ 提前收敛于第 {cycle + 1} 轮")
                    break
        
        print(f"\n{'='*60}")
        print("主动学习完成")
        print(f"{'='*60}")
        print(f"最终训练集大小: {len(self.train_points)}")
        print(f"总查询点数: {len(self.train_points)}")
        print(f"查询比例: {len(self.train_points)/self.Ln*100:.1f}%")
    
    def estimate_integral(self):
        """使用神经网络估计积分值"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(self.all_points)
                all_predictions.append(predictions)
        
        # 使用所有模型的平均预测
        avg_predictions = torch.mean(torch.stack(all_predictions), dim=0)
        
        # 蒙特卡洛积分
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        area = (x_max - x_min) * (y_max - y_min)
        
        integral = torch.mean(avg_predictions) * area
        
        return integral.item()
    
    def compute_exact_integral_mc(self, n_samples=100000):
        """使用蒙特卡洛方法计算精确积分（参考值）"""
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        
        x_samples = np.random.uniform(x_min, x_max, n_samples)
        y_samples = np.random.uniform(y_min, y_max, n_samples)
        
        values = []
        for x, y in zip(x_samples, y_samples):
            values.append(self.func(x, y))
        
        area = (x_max - x_min) * (y_max - y_min)
        integral = np.mean(values) * area
        
        return integral
    
    def get_final_predictions(self):
        """获取所有点的最终预测"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(self.all_points)
                all_predictions.append(predictions)
        
        predictions_tensor = torch.stack(all_predictions, dim=0)  # [5, Ln, 1]
        mean_predictions = predictions_tensor.mean(dim=0).squeeze()
        std_predictions = predictions_tensor.std(dim=0).squeeze()
        
        return {
            'points': self.all_points,
            'mean': mean_predictions,
            'std': std_predictions,
            'all_predictions': predictions_tensor
        }
    
    def plot_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. 训练损失曲线
        axes[0].plot(self.history['losses'], marker='o', linewidth=2)
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练损失曲线')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 积分估计收敛
        axes[1].plot(self.history['integral_estimates'], marker='s', linewidth=2, color='orange')
        if len(self.history['integral_estimates']) > 0:
            exact_integral = self.compute_exact_integral_mc(n_samples=50000)
            axes[1].axhline(y=exact_integral, color='r', linestyle='--', 
                           label=f'参考值: {exact_integral:.4f}')
        axes[1].set_xlabel('主动学习轮次')
        axes[1].set_ylabel('积分估计')
        axes[1].set_title('积分估计收敛')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 预测方差分布
        if self.history['variances']:
            final_variances = self.history['variances'][-1]
            axes[2].hist(final_variances, bins=50, alpha=0.7, color='green')
            axes[2].set_xlabel('预测方差')
            axes[2].set_ylabel('频数')
            axes[2].set_title('最终预测方差分布')
            axes[2].grid(True, alpha=0.3)
        
        # 4. 训练点分布
        train_points = self.train_points.numpy()
        axes[3].scatter(train_points[:, 0], train_points[:, 1], 
                       s=10, alpha=0.6, color='purple')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        axes[3].set_title(f'训练点分布 ({len(train_points)}个点)')
        axes[3].grid(True, alpha=0.3)
        
        # 5. 函数表面图（采样）
        if len(train_points) > 0:
            x_vals = train_points[:, 0]
            y_vals = train_points[:, 1]
            z_vals = self.train_values.numpy().flatten()
            
            sc = axes[4].scatter(x_vals, y_vals, c=z_vals, s=20, 
                                cmap='viridis', alpha=0.8)
            axes[4].set_xlabel('x')
            axes[4].set_ylabel('y')
            axes[4].set_title('函数值分布')
            plt.colorbar(sc, ax=axes[4])
            axes[4].grid(True, alpha=0.3)
        
        # 6. 各模型预测对比
        if hasattr(self, 'models') and len(self.models) > 0:
            predictions = []
            model_names = ['MLP', 'Fourier', 'ResNet', 'Siren', 'Wide']
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    sample_pred = model(self.all_points[:100])
                    predictions.append(sample_pred.mean().item())
            
            axes[5].bar(model_names, predictions, color=colors, alpha=0.7)
            axes[5].set_xlabel('模型')
            axes[5].set_ylabel('平均预测值')
            axes[5].set_title('各模型预测对比')
            axes[5].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


# ==================== 3. 测试函数定义 ====================

def test_function_1(x, y):
    """简单函数: sin(x)*cos(y) + 0.1*(x²+y²)"""
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

def test_function_2(x, y):
    """中等复杂度函数"""
    return np.sin(2*x + y) * np.cos(x - y) + 0.05 * np.exp(-0.1*(x**2 + y**2))

def test_function_3(x, y):
    """高频振荡函数"""
    return np.sin(10*x) * np.cos(8*y) + 0.5 * np.sin(5*x*y)

def test_function_4(x, y):
    """多峰函数"""
    return (np.sin(x) * np.cos(y) + 
            0.3 * np.exp(-((x-1)**2 + (y-1)**2)/0.2) +
            0.2 * np.exp(-((x+1)**2 + (y+1)**2)/0.3))

# ==================== 4. 主程序 ====================

def main():
    print("主动学习神经网络积分求解器")
    print("="*50)
    
    # 选择测试函数
    test_functions = {
        '1': test_function_1,
        '2': test_function_2,
        '3': test_function_3,
        '4': test_function_4
    }
    
    print("\n请选择测试函数:")
    print("1: sin(x)*cos(y) + 0.1*(x²+y²)")
    print("2: sin(2x+y)*cos(x-y) + 高斯项")
    print("3: 高频振荡函数")
    print("4: 多峰函数")
    
    choice = input("请输入选择 (1-4, 默认1): ").strip()
    if choice not in test_functions:
        choice = '1'
    
    target_func = test_functions[choice]
    
    # 定义积分域
    domain = [(-2, 2), (-2, 2)]  # x∈[-2,2], y∈[-2,2]
    
    print(f"\n积分域: x ∈ [{domain[0][0]}, {domain[0][1]}], y ∈ [{domain[1][0]}, {domain[1][1]}]")
    
    # 创建主动学习积分器
    print("\n创建主动学习积分器...")
    integrator = ActiveLearningIntegrator(
        target_function=target_func,
        domain=domain,
        Ln=2000,    # 总点数
        S0=100,     # 初始点数
        St=50,      # 每轮添加点数
        m=12        # 最大轮次
    )
    
    # 执行主动学习
    print("\n开始主动学习过程...")
    integrator.active_learning_cycle()
    
    # 计算最终结果
    final_integral = integrator.estimate_integral()
    reference_integral = integrator.compute_exact_integral_mc(n_samples=200000)
    
    print(f"\n{'='*60}")
    print("最终结果:")
    print(f"{'='*60}")
    print(f"最终积分估计: {final_integral:.6f}")
    print(f"参考积分值: {reference_integral:.6f}")
    print(f"绝对误差: {abs(final_integral - reference_integral):.6f}")
    print(f"相对误差: {abs(final_integral - reference_integral)/abs(reference_integral)*100:.2f}%")
    print(f"总查询点数: {len(integrator.train_points)}")
    print(f"查询比例: {len(integrator.train_points)/integrator.Ln*100:.1f}%")
    
    # 获取最终预测
    results = integrator.get_final_predictions()
    
    # 显示前5个点的预测
    print(f"\n前5个点的预测结果:")
    print("-"*60)
    for i in range(5):
        point = results['points'][i]
        pred_mean = results['mean'][i].item()
        pred_std = results['std'][i].item()
        exact = target_func(point[0].item(), point[1].item())
        
        print(f"点 {i+1}: ({point[0]:.3f}, {point[1]:.3f})")
        print(f"  预测值: {pred_mean:.6f} ± {pred_std:.6f}")
        print(f"  精确值: {exact:.6f}")
        print(f"  误差: {abs(pred_mean - exact):.6f}")
        print()
    
    # 可视化结果
    print("生成可视化图表...")
    integrator.plot_results()
    
    print(f"\n程序执行完成!")

# ==================== 5. 直接运行入口 ====================

if __name__ == "__main__":
    main()