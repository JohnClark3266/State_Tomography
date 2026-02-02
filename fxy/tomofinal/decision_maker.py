"""
采样决策模块

根据委员会预测结果决定下一轮采哪些点
完全独立于采样执行模块

职责:
- 分析委员会预测的不确定性
- 结合物理先验 (梯度、原点权重)
- 输出下一轮要采样的点索引
"""

import numpy as np
from typing import Tuple, Optional


class DecisionMaker:
    """采样决策器"""
    
    def __init__(self, grid_size: int = 64, 
                 variance_weight: float = 0.6,
                 gradient_weight: float = 0.3,
                 origin_weight: float = 0.1):
        """
        初始化决策器
        
        参数:
            grid_size: 网格大小
            variance_weight: 不确定性权重
            gradient_weight: 梯度权重
            origin_weight: 原点权重
        """
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        
        self.w_var = variance_weight
        self.w_grad = gradient_weight
        self.w_origin = origin_weight
        
        # 预计算原点权重图
        self._origin_weight_map = self._compute_origin_weight()
    
    def _compute_origin_weight(self) -> np.ndarray:
        """计算原点权重图"""
        center = self.grid_size // 2
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        sigma = self.grid_size / 4
        weight = np.exp(-dist**2 / (2 * sigma**2))
        return weight.flatten()
    
    def _compute_gradient(self, prediction: np.ndarray) -> np.ndarray:
        """计算预测的梯度幅值"""
        pred_2d = prediction.reshape(self.grid_size, self.grid_size)
        dy = np.abs(np.diff(pred_2d, axis=0, prepend=pred_2d[0:1, :]))
        dx = np.abs(np.diff(pred_2d, axis=1, prepend=pred_2d[:, 0:1]))
        gradient_mag = np.sqrt(dx**2 + dy**2)
        return gradient_mag.flatten()
    
    def decide_next_samples(self, 
                            variance: np.ndarray,
                            prediction: np.ndarray,
                            current_state: np.ndarray,
                            n_samples: int) -> np.ndarray:
        """
        决定下一轮采样的点
        
        参数:
            variance: 委员会预测方差 (N,) 或 (grid, grid)
            prediction: 委员会平均预测 (N,) 或 (grid, grid)
            current_state: 当前采样状态 (N,) - 0/1/2
            n_samples: 要选择的点数
        
        返回:
            indices: 要采样的点索引 (n_samples,)
        """
        # 展平
        if variance.ndim == 2:
            variance = variance.flatten()
        if prediction.ndim == 2:
            prediction = prediction.flatten()
        
        # 归一化各因素
        var_score = variance.copy()
        if var_score.max() > 0:
            var_score = var_score / var_score.max()
        
        grad_score = self._compute_gradient(prediction)
        if grad_score.max() > 0:
            grad_score = grad_score / grad_score.max()
        
        origin_score = self._origin_weight_map.copy()
        
        # 综合评分
        combined = (self.w_var * var_score + 
                    self.w_grad * grad_score + 
                    self.w_origin * origin_score)
        
        # [优化] 加入微小随机噪声以打破平局 (防止在无信息时退化为线性扫描)
        combined += np.random.normal(0, 1e-6, size=combined.shape)
        
        # 已采样或待采样的点不可选
        unavailable = (current_state != 0)  # 状态不为0的点不可选
        combined[unavailable] = -np.inf
        
        # 选择得分最高的n_samples个点
        n_available = np.sum(~unavailable)
        n_samples = min(n_samples, n_available)
        
        if n_samples <= 0:
            return np.array([], dtype=int)
        
        indices = np.argsort(combined)[-n_samples:]
        return indices
    
    def decide_initial_samples(self,
                               variance: np.ndarray,
                               prediction: np.ndarray,
                               n_samples: int) -> np.ndarray:
        """
        决定初始采样点 (所有点都可选)
        
        参数:
            variance: 委员会预测方差
            prediction: 委员会平均预测
            n_samples: 初始采样点数
        
        返回:
            indices: 初始采样点索引
        """
        # 初始状态全为0，都可选
        initial_state = np.zeros(self.n_points)
        return self.decide_next_samples(variance, prediction, initial_state, n_samples)
