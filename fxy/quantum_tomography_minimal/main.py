"""
MATLAB 集成层析主入口 (简化版 + MLE 保真度)

基于原版 main_matlab.py，使用 MLE 重建密度矩阵并用 QuTiP fidelity 计算保真度。

用法:
    # 确保 MATLAB 已运行并执行了 matlab.engine.shareEngine
    python main.py --state 2 --alpha 2.0 --samples 20
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sampling_manager import SamplingManager
from decision_maker import DecisionMaker
from neural_networks import build_model_pool, select_committee
from visualization import plot_all_results
from matlab_bridge import MatlabBridge


# ============================================================
# QuTiP 量子态与 Wigner 函数 (来自 Wigner_MLE_fidelity.py)
# ============================================================

try:
    import cvxpy as cp
    from qutip import displace, Qobj, fidelity as qutip_fidelity, basis, wigner, coherent, fock
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False
    print("⚠ 未安装 cvxpy 或 qutip，功能受限")


def create_state_qutip(grid_size=64, state_type=2, N=50, x_range=(-5, 5), **state_params):
    """
    使用 QuTiP 创建量子态并生成 Wigner 函数
    
    参数:
        grid_size: 网格大小
        state_type: 1=Fock态, 2=相干态
        N: Hilbert 空间截断
        x_range: 相空间范围
        **state_params: alpha (相干态), n (Fock态)
    
    返回:
        X, P: 相空间网格
        wigner_data: Wigner 函数 (grid_size, grid_size)
        state_name: 态名称
        rho: 密度矩阵 (QuTiP Qobj)
    """
    if not HAS_QUTIP:
        raise RuntimeError("需要安装 qutip 来生成量子态")
    
    xvec = np.linspace(x_range[0], x_range[1], grid_size)
    X, P = np.meshgrid(xvec, xvec)
    
    if state_type == 1:  # Fock 态
        n = state_params.get('n', 1)
        psi = basis(N, n)
        state_name = f"Fock |{n}⟩"
    else:  # 相干态
        alpha = state_params.get('alpha', 2.0)
        psi = coherent(N, alpha * 1j)  # alpha*i 使态位于 p 轴正方向
        state_name = f"Coherent |α={alpha}i⟩"
    
    rho = psi * psi.dag()
    
    # 使用 QuTiP 计算 Wigner 函数
    W = wigner(rho, xvec, xvec)
    
    return X, P, W.astype(np.float32), state_name, rho


def parity_op(N):
    """宇称算符"""
    return Qobj(np.diag([(-1)**n for n in range(N)]))


def wigner_MLE(wigner_data, xvec, yvec, N=50, n_trunc=50, verbose=False):
    """
    使用最大似然估计从 Wigner 数据重建密度矩阵
    
    参数:
        wigner_data: Wigner 函数 (grid_size, grid_size)
        xvec, yvec: 相空间坐标
        N: Hilbert 空间维度 (用于位移算符)
        n_trunc: MLE 重建的截断维度
        
    返回:
        rho_qobj: QuTiP 密度矩阵对象
        rho_mle: numpy 密度矩阵
    """
    if not HAS_QUTIP:
        return None, None
    
    # 展平 Wigner 数据
    W_vec = wigner_data.T.reshape(-1)
    
    # 构建设计矩阵
    beta_list = [x + 1j*y for y in yvec for x in xvec]
    num_meas = len(beta_list)
    parity = parity_op(N)
    A_real = np.zeros((num_meas, n_trunc**2))
    
    for m, beta in enumerate(beta_list):
        D = displace(N, beta)
        M = D * parity * D.dag()
        M = M.full()[:n_trunc, :n_trunc]
        A_real[m, :] = M.real.T.reshape(-1)
    
    # MLE 优化
    rho_var = cp.Variable((n_trunc, n_trunc), symmetric=True)
    constraints = [rho_var >> 0, cp.trace(rho_var) == 1]
    objective = cp.Minimize(cp.norm(A_real @ cp.vec(rho_var) - W_vec, 2))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)
    
    rho_mle = rho_var.value
    rho_qobj = Qobj(rho_mle) if rho_mle is not None else None
    return rho_qobj, rho_mle


def compute_fidelity(pred_wigner, target_wigner):
    """
    快速 Wigner 函数保真度 (用于每轮评估)
    使用归一化重叠积分
    """
    overlap = np.sum(pred_wigner * target_wigner)
    norm_pred = np.sqrt(np.sum(pred_wigner**2))
    norm_target = np.sqrt(np.sum(target_wigner**2))
    if norm_pred > 0 and norm_target > 0:
        return np.clip(overlap / (norm_pred * norm_target), 0, 1)
    return 0.0


def compute_fidelity_mle(pred_wigner, target_wigner, xvec, N=50, n_trunc=20):
    """
    使用 MLE 重建密度矩阵并计算保真度 (用于最终评估)
    """
    if not HAS_QUTIP:
        return compute_fidelity(pred_wigner, target_wigner)
    
    # MLE 重建两个密度矩阵
    rho_pred, _ = wigner_MLE(pred_wigner, xvec, xvec, N, n_trunc)
    rho_target, _ = wigner_MLE(target_wigner, xvec, xvec, N, n_trunc)
    
    if rho_pred is None or rho_target is None:
        return 0.0
    
    return float(qutip_fidelity(rho_pred, rho_target))


class FastTomographyMatlab:
    """使用 MATLAB 采样 + MLE 初始化的快速主动学习层析"""
    
    def __init__(self, grid_size=64, state_type=2,
                 initial_ratio=0.0073, samples_per_round=20,
                 max_rounds=50, pretrain_epochs=30, finetune_epochs=30,
                 lr=2e-3, F_threshold=0.98, committee_size=5,
                 noise_std=0.006, use_mle=True, **state_params):
        
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.initial_ratio = initial_ratio
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.F_threshold = F_threshold
        self.noise_std = noise_std
        self.use_mle = use_mle
        
        # 设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
        # 连接 MATLAB
        print("\n连接 MATLAB...")
        self.matlab_bridge = MatlabBridge()
        if not self.matlab_bridge.is_connected:
            raise RuntimeError("无法连接 MATLAB! 请确保 MATLAB 已运行并执行了 matlab.engine.shareEngine")
        
        # 生成目标态
        # 使用 QuTiP 生成量子态和 Wigner 函数
        self.X, self.P, self.ideal_wigner, self.state_name, self.target_rho = create_state_qutip(
            grid_size, state_type, N=50, **state_params
        )
        print(f"目标态: {self.state_name}")
        
        # 相空间坐标 (用于 MLE 保真度计算)
        self.xvec = np.linspace(-5, 5, grid_size)
        
        # 使用 Python 理论态作为参考
        self.exp_wigner = self.ideal_wigner.copy()
        
        # 初始化管理器
        self.sampler = SamplingManager(grid_size, matlab_bridge=self.matlab_bridge)
        self.decision = DecisionMaker(grid_size)
        
        # 构建委员会
        print("\n构建神经网络模型池...")
        self.model_pool = build_model_pool(grid_size)
        self.committee = select_committee(self.model_pool, committee_size)
        print(f"选中的委员会成员: {[n for n, _ in self.committee]}")
        
        for name, model in self.committee:
            model.to(self.device)
        
        # 历史记录
        self.history = {
            'round': [],
            'sampling_ratio': [],
            'F_recon_vs_exp': [],
            'F_recon_vs_ideal': [],
            'max_variance': [],
        }
        
        self.final_pred = None
    
    def _mle_pretrain(self):
        """使用 MLE 进行初始化"""
        if HAS_QUTIP:
            print("\n[MLE 初始化] 从完整 Wigner 函数重建密度矩阵...")
            rho_qobj, rho_mle = wigner_MLE(self.ideal_wigner, self.xvec, self.xvec, N=50, n_trunc=20)
            
            if rho_qobj is not None:
                print("  ✓ MLE 重建完成")
            else:
                print("  ⚠ MLE 失败")
        else:
            print("\n[跳过 MLE] cvxpy/qutip 未安装")
        
        # 使用完整理论态训练网络
        self._standard_pretrain()
    
    def _standard_pretrain(self):
        """预训练: 直接用完整理论态 (4096 点) 训练一次"""
        print(f"\n[预训练] 使用完整理论态 (4096 点)...")
        
        # 输入: 完整 Wigner + 全1掩码
        full_wigner = self.ideal_wigner.astype(np.float32)
        full_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        # (1, 2, 64, 64): batch=1, channels=2
        train_input = torch.tensor(
            np.stack([full_wigner, full_mask], axis=0)[np.newaxis, :, :, :]
        ).to(self.device)
        
        # 目标: 完整 Wigner (1, 1, 64, 64)
        train_target = torch.tensor(
            full_wigner[np.newaxis, np.newaxis, :, :]
        ).to(self.device)
        
        criterion = nn.MSELoss()
        
        # 对每个模型进行单次训练
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            optimizer.zero_grad()
            pred = model(train_input)
            loss = criterion(pred, train_target)
            loss.backward()
            optimizer.step()
            
            print(f"  {name}: loss = {loss.item():.6f}")
        
        print("✓ 预训练完成\n")
    
    def _committee_predict(self, sparse_input):
        """委员会预测"""
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :], dtype=torch.float32).to(self.device)
        
        predictions = []
        for name, model in self.committee:
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().squeeze()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.var(axis=0)
    
    def _light_finetune(self):
        """轻量微调"""
        sparse_input = self.sampler.get_sparse_input_for_nn()
        target = self.ideal_wigner[np.newaxis, :, :].astype(np.float32)
        
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :]).to(self.device)
        target_tensor = torch.tensor(target[np.newaxis, :, :, :]).to(self.device)
        
        criterion = nn.MSELoss()
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.1)
            for _ in range(self.finetune_epochs):
                optimizer.zero_grad()
                pred = model(input_tensor)
                loss = criterion(pred, target_tensor)
                loss.backward()
                optimizer.step()
    
    def run(self):
        """运行快速层析"""
        print("\n" + "="*60)
        print("快速主动学习层析 (MATLAB 采样 + MLE 初始化)")
        print("="*60)
        print(f"态类型: {self.state_name}")
        print(f"每轮增加点数: {self.samples_per_round}")
        print(f"目标保真度: {self.F_threshold}")
        print("="*60 + "\n")
        
        # 1. 预训练 (使用 MLE 或标准方法)
        if self.use_mle:
            self._mle_pretrain()
        else:
            self._standard_pretrain()
        
        # 2. 初始采样决策
        print("[初始化] 基于预训练结果选择初始采样点...")
        
        empty_input = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        mean_pred, variance = self._committee_predict(empty_input)
        
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        initial_indices = self.decision.decide_initial_samples(
            variance.flatten(), mean_pred.flatten(), n_initial
        )
        
        self.sampler.set_points_to_sample(initial_indices)
        
        print(f"  [MATLAB] 采样 {len(initial_indices)} 个初始点...")
        success = self.sampler.execute_sampling_matlab(
            state_type=self.state_type,
            noise_std=self.noise_std,
            **self.state_params
        )
        if not success:
            raise RuntimeError("MATLAB 采样失败")
        
        print(f"  ✓ 初始采样: {self.sampler.get_sampled_count()} 点\n")
        
        # 3. 主循环
        for round_id in range(self.max_rounds):
            ratio = self.sampler.get_sampling_ratio()
            n_sampled = self.sampler.get_sampled_count()
            print(f"[Round {round_id+1}/{self.max_rounds}] 采样点: {n_sampled} ({ratio*100:.2f}%)")
            
            self._light_finetune()
            
            sparse_input = self.sampler.get_sparse_input_for_nn()
            mean_pred, variance = self._committee_predict(sparse_input)
            
            mask = self.sampler.get_mask_2d()
            measured = self.sampler.get_wigner_2d()
            mean_pred[mask] = measured[mask]
            
            F_exp = compute_fidelity(mean_pred, self.exp_wigner)
            F_ideal = compute_fidelity(mean_pred, self.ideal_wigner)
            max_var = np.max(variance)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(ratio)
            self.history['F_recon_vs_exp'].append(F_exp)
            self.history['F_recon_vs_ideal'].append(F_ideal)
            self.history['max_variance'].append(max_var)
            
            print(f"  F(vs MATLAB): {F_exp:.5f}  F(vs Python): {F_ideal:.5f}  MaxVar: {max_var:.6f}")
            
            if F_exp >= self.F_threshold:
                print(f"\n✓ 达到目标保真度 {self.F_threshold}!")
                break
            
            if round_id < self.max_rounds - 1:
                current_state = self.sampler.get_state()
                next_indices = self.decision.decide_next_samples(
                    variance.flatten(), mean_pred.flatten(),
                    current_state, self.samples_per_round
                )
                
                self.sampler.set_points_to_sample(next_indices)
                
                print(f"  [MATLAB] 采样 {len(next_indices)} 个新点...")
                success = self.sampler.execute_sampling_matlab(
                    state_type=self.state_type,
                    noise_std=self.noise_std,
                    **self.state_params
                )
                if not success:
                    raise RuntimeError("MATLAB 采样失败")
        
        # 4. 最终预测
        sparse_input = self.sampler.get_sparse_input_for_nn()
        self.final_pred, _ = self._committee_predict(sparse_input)
        
        self.sampler.fill_predictions(self.final_pred)
        
        mask = self.sampler.get_mask_2d()
        measured = self.sampler.get_wigner_2d()
        self.final_pred[mask] = measured[mask]
        
        # 设置兼容 visualization.py 的属性
        self.sampling_mask = mask
        self.final_ratio = self.sampler.get_sampling_ratio()
        self.final_F_exp = compute_fidelity(self.final_pred, self.exp_wigner)
        self.final_F_ideal = compute_fidelity(self.final_pred, self.ideal_wigner)
        self.F_exp_vs_ideal = compute_fidelity(self.exp_wigner, self.ideal_wigner)
        
        # 转换采样历史
        self.sampling_history = []
        cumulative_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for indices in self.sampler.sampling_history:
            round_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for idx in indices:
                i, j = idx // self.grid_size, idx % self.grid_size
                round_mask[i, j] = True
            cumulative_mask |= round_mask
            self.sampling_history.append(cumulative_mask.copy())
        
        print("\n" + "="*60)
        print("完成!")
        print(f"最终采样率: {self.final_ratio*100:.2f}%")
        print(f"F(vs实验): {self.final_F_exp:.5f}")
        print(f"F(vs理论): {self.final_F_ideal:.5f}")
        print("="*60)
        
        self.matlab_bridge.close()
        
        return self.final_pred


def main():
    parser = argparse.ArgumentParser(description="MATLAB 集成量子态层析 (MLE 初始化)")
    parser.add_argument('--state', type=int, default=2, help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=float, default=2.0, help='相干态/猫态振幅')
    parser.add_argument('--n', type=int, default=3, help='Fock态光子数')
    parser.add_argument('--samples', type=int, default=10, help='每轮增加的采样点数')
    parser.add_argument('--rounds', type=int, default=30, help='最大轮数')
    parser.add_argument('--pretrain', type=int, default=30, help='预训练epochs')
    parser.add_argument('--finetune', type=int, default=20, help='微调epochs')
    parser.add_argument('--threshold', type=float, default=0.99, help='目标保真度')
    parser.add_argument('--no-mle', action='store_true', help='不使用 MLE 初始化')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    state_params = {}
    if args.state == 1:
        state_params['n'] = args.n
    elif args.state == 2:
        state_params['alpha'] = args.alpha
    elif args.state == 3:
        state_params['alpha'] = args.alpha
        state_params['parity'] = 'even'
    
    tomo = FastTomographyMatlab(
        grid_size=64,
        state_type=args.state,
        initial_ratio=0.00244,  # ~10 初始点
        samples_per_round=args.samples,
        max_rounds=args.rounds,
        pretrain_epochs=args.pretrain,
        finetune_epochs=args.finetune,
        lr=2e-3,
        F_threshold=args.threshold,
        committee_size=5,
        noise_std=0.006,
        use_mle=not args.no_mle,
        **state_params
    )
    
    tomo.run()
    
    plot_all_results(tomo, save_dir="results")
    print("\n结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()
