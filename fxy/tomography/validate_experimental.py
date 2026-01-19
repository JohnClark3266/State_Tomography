"""
使用模拟实验数据验证训练好的CNN委员会

生成带有实验噪声的GKP态数据，并用5个训练好的CNN进行重建
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# 导入训练模块
from cnn_gkp_training import (
    CNN1, CNN2, CNN3, CNN4, CNN5,
    gkp_wigner, generate_gkp_image
)

# 设置随机种子
np.random.seed(2024)
torch.manual_seed(2024)


def generate_experimental_gkp(grid_size=64, delta=0.32, noise_type='gaussian'):
    """
    生成模拟实验测量的GKP态数据
    
    模拟实验中常见的误差源:
    1. 高斯噪声 (探测器热噪声)
    2. 泊松噪声 (光子计数统计涨落)
    3. 系统偏移 (校准误差)
    4. 部分测量 (有限采样)
    
    参数:
        grid_size: 图像分辨率
        delta: GKP态压缩参数
        noise_type: 噪声类型 ('gaussian', 'poisson', 'mixed')
    
    返回:
        (noisy_data, clean_data, params)
    """
    x = np.linspace(-4, 4, grid_size)
    p = np.linspace(-4, 4, grid_size)
    X, P = np.meshgrid(x, p)
    
    # 生成干净的GKP态
    clean = gkp_wigner(X, P, delta=delta)
    
    # 添加实验噪声
    noisy = clean.copy()
    
    if noise_type == 'gaussian' or noise_type == 'mixed':
        # 热噪声: 高斯分布
        gaussian_noise = np.random.normal(0, 0.08 * np.abs(clean).max(), clean.shape)
        noisy += gaussian_noise
    
    if noise_type == 'poisson' or noise_type == 'mixed':
        # 光子计数噪声: 使用泊松近似
        # 将Wigner函数映射到正值范围再添加泊松噪声
        min_val = noisy.min()
        shifted = noisy - min_val + 0.1
        poisson_noise = np.random.poisson(shifted * 10) / 10 - shifted
        noisy += poisson_noise * 0.5
    
    # 系统偏移 (模拟校准误差)
    offset = np.random.uniform(-0.02, 0.02)
    noisy += offset
    
    # 轻微的空间模糊 (模拟光学系统的有限分辨率)
    from scipy.ndimage import gaussian_filter
    noisy = gaussian_filter(noisy, sigma=0.5)
    
    params = {
        'delta': delta,
        'noise_type': noise_type,
        'offset': offset,
    }
    
    return noisy.astype(np.float32), clean.astype(np.float32), params


def load_trained_models():
    """加载并初始化训练好的模型架构"""
    # 注意: 由于没有保存模型权重，我们需要重新训练
    # 这里我们创建新的模型实例并快速训练
    
    print("正在准备CNN委员会模型...")
    
    models = [
        ("CNN1_ReLU_3Layer", CNN1()),
        ("CNN2_Tanh_4Layer", CNN2()),
        ("CNN3_LeakyReLU_Wide", CNN3()),
        ("CNN4_ResNet", CNN4()),
        ("CNN5_UNet", CNN5()),
    ]
    
    # 快速训练 (使用少量数据)
    from cnn_gkp_training import generate_training_data
    from torch.utils.data import DataLoader, TensorDataset
    
    print("快速训练模型 (少量epoch用于演示)...")
    inputs, targets = generate_training_data(n_samples=200, grid_size=64)
    
    train_inputs = torch.tensor(inputs, dtype=torch.float32)
    train_targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(train_inputs, train_targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    criterion = torch.nn.MSELoss()
    
    for name, model in models:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(30):  # 快速训练30个epoch
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        
        print(f"  {name} 训练完成")
    
    return models, device


def validate_with_experimental_data():
    """使用模拟实验数据验证CNN委员会"""
    
    print("="*60)
    print("CNN委员会验证 - 使用模拟实验GKP态数据")
    print("="*60)
    
    # 加载模型
    models, device = load_trained_models()
    
    # 生成多组模拟实验数据
    test_cases = [
        {'delta': 0.30, 'noise_type': 'gaussian'},
        {'delta': 0.35, 'noise_type': 'poisson'},
        {'delta': 0.28, 'noise_type': 'mixed'},
    ]
    
    fig, axes = plt.subplots(len(test_cases), 8, figsize=(24, 3*len(test_cases)))
    
    print("\n开始验证...")
    
    all_results = []
    
    for row, params in enumerate(test_cases):
        noisy, clean, info = generate_experimental_gkp(**params)
        
        # 准备输入
        input_tensor = torch.tensor(noisy[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        input_tensor = input_tensor.to(device)
        
        # 通过每个模型预测
        predictions = []
        mse_scores = []
        
        for name, model in models:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append((name, pred))
            mse = np.mean((pred - clean)**2)
            mse_scores.append(mse)
        
        all_results.append({
            'params': info,
            'predictions': predictions,
            'mse_scores': mse_scores,
            'clean': clean,
            'noisy': noisy,
        })
        
        # 绘图
        # 1. 输入 (噪声)
        axes[row, 0].imshow(noisy, cmap='RdBu_r', origin='lower')
        axes[row, 0].set_title(f"Input\n({info['noise_type']} noise)")
        axes[row, 0].axis('off')
        
        # 2. 目标 (干净)
        axes[row, 1].imshow(clean, cmap='RdBu_r', origin='lower')
        axes[row, 1].set_title(f"Target\n(δ={info['delta']})")
        axes[row, 1].axis('off')
        
        # 3-7. 各模型预测
        for i, ((name, pred), mse) in enumerate(zip(predictions, mse_scores)):
            axes[row, i+2].imshow(pred, cmap='RdBu_r', origin='lower')
            short_name = name.split('_')[0]
            axes[row, i+2].set_title(f"{short_name}\nMSE:{mse:.5f}")
            axes[row, i+2].axis('off')
        
        # 8. 最佳模型的误差图
        best_idx = np.argmin(mse_scores)
        best_pred = predictions[best_idx][1]
        error = np.abs(best_pred - clean)
        im = axes[row, 7].imshow(error, cmap='hot', origin='lower')
        axes[row, 7].set_title(f"Error\n(Best: {predictions[best_idx][0].split('_')[0]})")
        axes[row, 7].axis('off')
    
    plt.suptitle("CNN Committee Validation on Simulated Experimental GKP Data", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experimental_validation.png', dpi=300, bbox_inches='tight')
    print("\n保存验证结果图到 tomography/experimental_validation.png")
    plt.close()
    
    # 打印统计结果
    print("\n" + "="*60)
    print("验证结果统计")
    print("="*60)
    
    model_names = [name for name, _ in models]
    avg_mse = np.zeros(len(models))
    
    for result in all_results:
        avg_mse += np.array(result['mse_scores'])
    avg_mse /= len(all_results)
    
    print("\n各模型平均MSE:")
    for name, mse in sorted(zip(model_names, avg_mse), key=lambda x: x[1]):
        marker = "⭐" if mse == avg_mse.min() else "  "
        print(f"  {marker} {name}: {mse:.6f}")
    
    print("\n" + "="*60)
    print("验证完成!")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    validate_with_experimental_data()
