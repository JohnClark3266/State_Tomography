yongimport numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 1. 目标二维函数: Fock |3⟩ 态 Wigner 函数
# =========================

def fock_wigner(x, p, n=3):
    """
    Fock态 |n⟩ 的Wigner函数
    
    W_n(x, p) = ((-1)^n / π) * L_n(2*(x^2 + p^2)) * exp(-(x^2 + p^2))
    
    其中 L_n 是Laguerre多项式
    
    参数:
        x: 位置坐标 (相空间)
        p: 动量坐标 (相空间)
        n: Fock态的光子数 (默认n=3)
    
    返回:
        Wigner函数值
    """
    r_squared = x**2 + p**2
    L_n = genlaguerre(n, 0)
    laguerre_val = L_n(2 * r_squared)
    wigner = ((-1)**n / np.pi) * laguerre_val * np.exp(-r_squared)
    return wigner

def target_u(x, y):
    """Fock |3⟩ 态的Wigner函数 - 量子态相空间分布"""
    return fock_wigner(x, y, n=3)


# =========================
# 2. 委员会模型定义
# =========================

class MLP(nn.Module):
    def __init__(self, layers, activation="tanh"):
        super().__init__()
        acts = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }
        act = acts[activation]
        net = []
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            net.append(nn.Linear(in_dim, out_dim))
            net.append(act())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def build_committee(n_init=3):
    configs = [
        ([2, 64, 64, 1], "tanh"),
        ([2, 64, 32, 1], "relu"),
        ([2, 80, 40, 1], "tanh"),
        ([2, 128, 64, 1], "relu"),
        ([2, 64, 64, 32, 1], "tanh"),
    ]
    committee = []
    for layers, act in configs:
        for _ in range(n_init):
            committee.append(MLP(layers, activation=act))
    return committee


# =========================
# 3. 主动学习流程
# =========================

class ActiveLearner2D:
    def __init__(
        self,
        grid_size=80,
        S0=120,
        St=60,
        rounds=10,
        n_init=3,
        lr=1e-3,
        epochs=200,
    ):
        self.grid_size = grid_size
        self.S0 = S0
        self.St = St
        self.rounds = rounds
        self.lr = lr
        self.epochs = epochs

        # 构建网格 - 相空间范围 [-4, 4]
        self.x = np.linspace(-4, 4, grid_size)
        self.y = np.linspace(-4, 4, grid_size)
        X, Y = np.meshgrid(self.x, self.y)
        self.points = np.stack([X.ravel(), Y.ravel()], axis=1)
        self.true_values = target_u(self.points[:, 0], self.points[:, 1]).reshape(-1, 1)

        self.points_t = torch.tensor(self.points, dtype=torch.float32)
        self.true_t = torch.tensor(self.true_values, dtype=torch.float32)

        # 委员会模型
        self.committee = build_committee(n_init=n_init)

        # 记录
        self.selected_indices = []
        self.snapshots = {}

        # 已标注集合
        self.labeled_mask = np.zeros(len(self.points), dtype=bool)

    def _train_single(self, model, dataset):
        model.train()
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = np.inf
        patience = 12
        patience_count = 0

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

    def _train_committee(self, x_train, y_train):
        dataset = TensorDataset(x_train, y_train)
        for model in self.committee:
            self._train_single(model, dataset)

    def _committee_predictions(self):
        preds = []
        for model in self.committee:
            model.eval()
            with torch.no_grad():
                preds.append(model(self.points_t))
        preds = torch.stack(preds, dim=0)  # [M, N, 1]
        mean = preds.mean(dim=0).squeeze(1)
        var = preds.var(dim=0).squeeze(1)
        return mean.numpy(), var.numpy()

    def _select_new_points(self, variances):
        candidate_variances = variances.copy()
        candidate_variances[self.labeled_mask] = -np.inf
        new_idx = np.argsort(candidate_variances)[-self.St :]
        return new_idx

    def run(self):
        # 初始采样
        init_idx = np.random.choice(len(self.points), self.S0, replace=False)
        self.labeled_mask[init_idx] = True
        self.selected_indices.extend(init_idx.tolist())

        for round_id in range(self.rounds):
            print(f"[Round {round_id + 1}/{self.rounds}] Train models...")
            x_train = self.points_t[self.labeled_mask]
            y_train = self.true_t[self.labeled_mask]
            self._train_committee(x_train, y_train)

            mean_pred, variances = self._committee_predictions()

            # 保存快照（早期与末期）
            if round_id in (1, self.rounds - 1):
                self.snapshots[round_id] = mean_pred

            new_idx = self._select_new_points(variances)
            self.labeled_mask[new_idx] = True
            self.selected_indices.extend(new_idx.tolist())

            print(
                f"  labeled: {self.labeled_mask.sum()} / {len(self.points)} "
                f"({self.labeled_mask.sum()/len(self.points)*100:.1f}%)"
            )

        # 最终预测
        final_pred, _ = self._committee_predictions()
        self.final_pred = final_pred

    def plot_results(self):
        X, Y = np.meshgrid(self.x, self.y)
        U_true = self.true_values.reshape(self.grid_size, self.grid_size)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) 真实Wigner函数
        im0 = axes[0, 0].contourf(X, Y, U_true, levels=40, cmap="RdBu_r")
        axes[0, 0].set_title("(a) Fock |3⟩ Wigner Function")
        axes[0, 0].set_xlabel("x (位置)")
        axes[0, 0].set_ylabel("p (动量)")
        fig.colorbar(im0, ax=axes[0, 0])

        # (b) 查询点分布
        axes[0, 1].contourf(X, Y, U_true, levels=30, cmap="RdBu_r")
        pts = self.points[self.selected_indices]
        axes[0, 1].scatter(pts[:, 0], pts[:, 1], s=10, c="cyan", edgecolor="k", lw=0.3)
        axes[0, 1].set_title("(b) Queried Points Distribution")
        axes[0, 1].set_xlabel("x (位置)")
        axes[0, 1].set_ylabel("p (动量)")

        # (c) 中期误差
        early_round = 1
        if early_round in self.snapshots:
            err_early = (
                self.snapshots[early_round] - self.true_values.squeeze()
            ).reshape(self.grid_size, self.grid_size)
        else:
            err_early = (self.final_pred - self.true_values.squeeze()).reshape(
                self.grid_size, self.grid_size
            )
        im2 = axes[1, 0].contourf(X, Y, err_early, levels=40, cmap="coolwarm")
        axes[1, 0].set_title("(c) Absolute Error (Early)")
        axes[1, 0].set_xlabel("x (位置)")
        axes[1, 0].set_ylabel("p (动量)")
        fig.colorbar(im2, ax=axes[1, 0])

        # (d) 末期误差
        err_final = (self.final_pred - self.true_values.squeeze()).reshape(
            self.grid_size, self.grid_size
        )
        im3 = axes[1, 1].contourf(X, Y, err_final, levels=40, cmap="coolwarm")
        axes[1, 1].set_title("(d) Absolute Error (Final)")
        axes[1, 1].set_xlabel("x (位置)")
        axes[1, 1].set_ylabel("p (动量)")
        fig.colorbar(im3, ax=axes[1, 1])

        plt.suptitle("Fock |3⟩ State Wigner Function - Active Learning Results", fontsize=14, fontweight='bold')
        plt.tight_layout()
        # plt.show()
        plt.savefig('ppt/images/wigner_results.png', dpi=300)
        print("已保存 Wigner 结果图到 ppt/images/wigner_results.png")
        
        # 新增: 绘制相对误差百分比图
        self.plot_relative_error()
    
    def plot_relative_error(self):
        """绘制每个点的相对误差百分比分布"""
        X, Y = np.meshgrid(self.x, self.y)
        
        # 计算相对误差百分比
        true_vals = self.true_values.squeeze()
        pred_vals = self.final_pred
        
        # 避免除零: |pred - true| / (|true| + epsilon) * 100
        epsilon = 1e-10
        absolute_error = np.abs(pred_vals - true_vals)
        relative_error_percent = absolute_error / (np.abs(true_vals) + epsilon) * 100
        
        # 限制最大显示为500%
        relative_error_clipped = np.clip(relative_error_percent, 0, 500)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 相对误差分布直方图
        axes[0].hist(relative_error_clipped, bins=50, alpha=0.7, color='crimson', edgecolor='black')
        axes[0].set_xlabel('相对误差 (%)')
        axes[0].set_ylabel('点数')
        axes[0].set_title('相对误差百分比分布')
        axes[0].axvline(x=np.median(relative_error_percent), color='blue', linestyle='--', 
                       label=f'中位数: {np.median(relative_error_percent):.1f}%')
        axes[0].axvline(x=np.mean(relative_error_percent), color='green', linestyle='--', 
                       label=f'均值: {np.mean(relative_error_percent):.1f}%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 相对误差空间分布热力图
        err_map = relative_error_clipped.reshape(self.grid_size, self.grid_size)
        im = axes[1].contourf(X, Y, err_map, levels=40, cmap='hot')
        axes[1].set_xlabel('x (位置)')
        axes[1].set_ylabel('p (动量)')
        axes[1].set_title('相对误差空间分布 (%)')
        plt.colorbar(im, ax=axes[1], label='相对误差 (%)')
        
        # 3. 预测值 vs 精确值 散点图
        axes[2].scatter(true_vals, pred_vals, s=3, alpha=0.4, color='steelblue')
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
        axes[2].set_xlabel('精确值')
        axes[2].set_ylabel('预测值')
        axes[2].set_title('预测值 vs 精确值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_aspect('equal', 'box')
        
        plt.suptitle(f'模型预测相对误差分析\n'
                    f'平均相对误差: {np.mean(relative_error_percent):.2f}% | '
                    f'中位数: {np.median(relative_error_percent):.2f}% | '
                    f'误差<10%点占比: {np.mean(relative_error_percent < 10)*100:.1f}%', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        # plt.show()
        plt.savefig('ppt/images/error_analysis.png', dpi=300)
        print("已保存误差分析图到 ppt/images/error_analysis.png")
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print("相对误差统计:")
        print(f"{'='*60}")
        print(f"平均相对误差: {np.mean(relative_error_percent):.2f}%")
        print(f"中位数相对误差: {np.median(relative_error_percent):.2f}%")
        print(f"最大相对误差: {np.max(relative_error_percent):.2f}%")
        print(f"最小相对误差: {np.min(relative_error_percent):.2f}%")
        print(f"相对误差 < 5% 的点占比: {np.mean(relative_error_percent < 5)*100:.1f}%")
        print(f"相对误差 < 10% 的点占比: {np.mean(relative_error_percent < 10)*100:.1f}%")
        print(f"相对误差 < 20% 的点占比: {np.mean(relative_error_percent < 20)*100:.1f}%")


def main():
    print("="*60)
    print("主动学习神经网络 - Fock |3⟩ 态 Wigner 函数拟合")
    print("="*60)
    print("目标函数: Fock |3⟩ 量子态的Wigner函数")
    print("相空间范围: x ∈ [-4, 4], p ∈ [-4, 4]")
    print("="*60 + "\n")
    
    learner = ActiveLearner2D(
        grid_size=80,
        S0=120,
        St=60,
        rounds=10,
        n_init=3,
        lr=1e-3,
        epochs=200,
    )
    learner.run()
    learner.plot_results()
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    # Ensure image directory exists
    import os
    os.makedirs('ppt/images', exist_ok=True)
    main()
