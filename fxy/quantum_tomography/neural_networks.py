"""
神经网络模型池模块

提供20个不同架构的神经网络模型，每次训练随机选择5个组成委员会。
"""

import torch
import torch.nn as nn
import numpy as np
import random


# ==========================================
# 模型架构1-5: 3层CNN (ReLU)，不同通道数
# ==========================================

class CNN_ReLU_Small(nn.Module):
    """3层CNN，小通道 (16-32-16)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_ReLU_Medium(nn.Module):
    """3层CNN，中等通道 (32-64-32)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_ReLU_Large(nn.Module):
    """3层CNN，大通道 (64-128-64)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_ReLU_5x5(nn.Module):
    """3层CNN，5x5卷积核"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_ReLU_Deep(nn.Module):
    """5层CNN，ReLU"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 模型架构6-8: 4层CNN (Tanh)
# ==========================================

class CNN_Tanh_Narrow(nn.Module):
    """4层CNN，Tanh激活，窄通道"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, 3, padding=1), nn.Tanh(),
            nn.Conv2d(32, 16, 3, padding=1), nn.Tanh(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_Tanh_Wide(nn.Module):
    """4层CNN，Tanh激活，宽通道"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 48, 3, padding=1), nn.Tanh(),
            nn.Conv2d(48, 96, 3, padding=1), nn.Tanh(),
            nn.Conv2d(96, 48, 3, padding=1), nn.Tanh(),
            nn.Conv2d(48, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_Tanh_Deep(nn.Module):
    """6层CNN，Tanh激活"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.Tanh(),
            nn.Conv2d(32, 64, 3, padding=1), nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1), nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1), nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1), nn.Tanh(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 模型架构9-12: CNN + BatchNorm
# ==========================================

class CNN_BN_Small(nn.Module):
    """CNN + BatchNorm，小型"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_BN_Large(nn.Module):
    """CNN + BatchNorm，大型"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_BN_LeakyReLU(nn.Module):
    """CNN + BatchNorm + LeakyReLU"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 48, 3, padding=1), nn.BatchNorm2d(48), nn.LeakyReLU(0.2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.LeakyReLU(0.2),
            nn.Conv2d(96, 48, 3, padding=1), nn.BatchNorm2d(48), nn.LeakyReLU(0.2),
            nn.Conv2d(48, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


class CNN_BN_ELU(nn.Module):
    """CNN + BatchNorm + ELU"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)


# ==========================================
# 模型架构13-16: ResNet风格
# ==========================================

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


class ResNet_2Block(nn.Module):
    """ResNet，2个残差块"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 32, 3, padding=1)
        self.block1 = ResBlock(32)
        self.block2 = ResBlock(32)
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.conv_out(x)


class ResNet_3Block(nn.Module):
    """ResNet，3个残差块"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 48, 3, padding=1)
        self.block1 = ResBlock(48)
        self.block2 = ResBlock(48)
        self.block3 = ResBlock(48)
        self.conv_out = nn.Conv2d(48, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.conv_out(x)


class ResNet_4Block(nn.Module):
    """ResNet，4个残差块"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 64, 3, padding=1)
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.block3 = ResBlock(64)
        self.block4 = ResBlock(64)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.conv_out(x)


class ResNet_Wide(nn.Module):
    """宽ResNet，2个残差块，更多通道"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 96, 3, padding=1)
        self.block1 = ResBlock(96)
        self.block2 = ResBlock(96)
        self.conv_out = nn.Conv2d(96, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.conv_out(x)


# ==========================================
# 模型架构17-20: 全连接网络
# ==========================================

class FC_Small(nn.Module):
    """全连接网络，小型 (适用于64x64)"""
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * grid_size * grid_size, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, grid_size * grid_size)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        out = self.fc(x)
        return out.view(batch, 1, self.grid_size, self.grid_size)


class FC_Medium(nn.Module):
    """全连接网络，中型"""
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * grid_size * grid_size, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, grid_size * grid_size)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        out = self.fc(x)
        return out.view(batch, 1, self.grid_size, self.grid_size)


class FC_Tanh(nn.Module):
    """全连接网络，Tanh激活"""
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * grid_size * grid_size, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, grid_size * grid_size)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        out = self.fc(x)
        return out.view(batch, 1, self.grid_size, self.grid_size)


class FC_LeakyReLU(nn.Module):
    """全连接网络，LeakyReLU激活"""
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * grid_size * grid_size, 1536), nn.LeakyReLU(0.1),
            nn.Linear(1536, 768), nn.LeakyReLU(0.1),
            nn.Linear(768, grid_size * grid_size)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        out = self.fc(x)
        return out.view(batch, 1, self.grid_size, self.grid_size)


def build_model_pool(grid_size=64):
    """
    构建20个不同架构的神经网络模型池
    (移除小模型，保留中大型模型以提升能力)
    
    返回:
        list: [(模型名称, 模型实例), ...]
    """
    return [
        # --- CNN ReLU ---
        ("CNN_ReLU_Medium", CNN_ReLU_Medium()),
        ("CNN_ReLU_Large", CNN_ReLU_Large()),
        ("CNN_ReLU_Deep", CNN_ReLU_Deep()),
        ("CNN_ReLU_5x5", CNN_ReLU_5x5()),
        
        # --- CNN Tanh ---
        ("CNN_Tanh_Wide", CNN_Tanh_Wide()),
        ("CNN_Tanh_Deep", CNN_Tanh_Deep()),
        
        # --- CNN BN ---
        ("CNN_BN_Large", CNN_BN_Large()),
        ("CNN_BN_LeakyReLU", CNN_BN_LeakyReLU()),
        ("CNN_BN_ELU", CNN_BN_ELU()),
        
        # --- ResNet ---
        ("ResNet_3Block", ResNet_3Block()),
        ("ResNet_4Block", ResNet_4Block()),
        ("ResNet_Wide", ResNet_Wide()),
        
        # --- FC (全连接) ---
        ("FC_Medium", FC_Medium(grid_size)),
        ("FC_Tanh", FC_Tanh(grid_size)),
        ("FC_LeakyReLU", FC_LeakyReLU(grid_size)),
        
        # --- 重复强模型以凑够数量或增加选中概率 ---
        ("ResNet_Wide_2", ResNet_Wide()),
        ("CNN_BN_Large_2", CNN_BN_Large()),
        ("CNN_ReLU_Deep_2", CNN_ReLU_Deep()),
        ("CNN_Tanh_Deep_2", CNN_Tanh_Deep()),
        ("ResNet_4Block_2", ResNet_4Block()),
    ]


def select_committee(pool, n=5, seed=None):
    """
    从模型池中随机选择n个模型组成委员会
    
    参数:
        pool: 模型池 (build_model_pool的返回值)
        n: 委员会成员数量 (默认5)
        seed: 随机种子 (可选)
    
    返回:
        list: [(模型名称, 模型实例), ...]
    """
    if seed is not None:
        random.seed(seed)
    
    selected = random.sample(pool, n)
    return selected
