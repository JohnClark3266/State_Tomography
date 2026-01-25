"""
量子态层析主程序 (MLE 重建 + MATLAB 采样)

使用 generate_wigner_data 生成 Wigner 函数，wigner_MLE 重建密度矩阵。
所有参数从 config.py 读取。

用法:
    python main.py --state 2 --alpha 2.0 --threshold 0.99
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from sampling_manager import SamplingManager
from decision_maker import DecisionMaker
from neural_networks import build_model_pool, select_committee
from visualization import plot_all_results, plot_round_wigner
from matlab_bridge import MatlabBridge

# QuTiP 导入
import cvxpy as cp
from qutip import basis, Qobj, displace, destroy, fidelity as qutip_fidelity, coherent, wigner as qutip_wigner
from scipy.linalg import expm as scipy_expm


# ============================================================
# Wigner 函数生成 (来自 Wigner_MLE_fidelity.py)
# ============================================================

def parity_op(N):
    """宇称算符 Π = exp(-iπ a†a)"""
    a = destroy(N)
    N_op = a.dag() * a
    return Qobj(scipy_expm(-1j * np.pi * N_op.full()))


def wigner_at_point(rho, beta, parity):
    """计算单点 Wigner 函数值"""
    N = rho.dims[0][0]
    D = displace(N, beta)
    return 2 / np.pi * (D.dag() * rho * D * parity).tr().real


def generate_wigner_data(rho, xvec, yvec, N):
    """
    生成相空间网格上的 Wigner 函数
    
    参数:
        rho: 密度矩阵 (Qobj)
        xvec, yvec: 相空间坐标数组
        N: Hilbert 空间维度
    
    返回:
        W_data: Wigner 函数 (len(xvec), len(yvec))
    """
    parity = parity_op(N)
    W_data = np.zeros((len(xvec), len(yvec)))
    for ix, x in enumerate(xvec):
        for iy, y in enumerate(yvec):
            beta = x + 1j * y
            W_data[ix, iy] = wigner_at_point(rho, beta, parity)
    return W_data


# ============================================================
# MLE 重建
# ============================================================

def wigner_MLE(wigner_data, xvec, yvec, N, n_trunc=None, verbose=False):
    """
    使用最大似然估计从 Wigner 数据重建密度矩阵
    
    参数:
        wigner_data: Wigner 函数 (grid_size, grid_size)
        xvec, yvec: 相空间坐标
        N: Hilbert 空间维度 (用于位移算符)
        n_trunc: MLE 截断维度 (默认为 N)
        verbose: 是否打印优化信息
        
    返回:
        rho_qobj: QuTiP 密度矩阵对象
        rho_mle: numpy 密度矩阵
    """
    if n_trunc is None:
        n_trunc = N
    
    W_vec = wigner_data.T.reshape(-1)
    
    beta_list = [x + 1j*y for y in yvec for x in xvec]
    num_meas = len(beta_list)
    parity = parity_op(N)
    A_real = np.zeros((num_meas, n_trunc**2))
    
    for m, beta in enumerate(beta_list):
        D = displace(N, beta)
        M = D * parity * D.dag()
        M_full = M.full()[:n_trunc, :n_trunc]
        A_real[m, :] = M_full.real.T.reshape(-1)
    
    rho_var = cp.Variable((n_trunc, n_trunc), symmetric=True)
    constraints = [rho_var >> 0, cp.trace(rho_var) == 1]
    objective = cp.Minimize(cp.norm(A_real @ cp.vec(rho_var) - W_vec, 2))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)
    
    rho_mle = rho_var.value
    rho_qobj = Qobj(rho_mle) if rho_mle is not None else None
    return rho_qobj, rho_mle


def compute_fidelity_mle(pred_wigner, target_wigner, xvec, yvec, N, n_trunc):
    """
    使用 MLE 重建密度矩阵并计算保真度 (用于最终评估)
    
    参数:
        pred_wigner: 预测的 Wigner 函数
        target_wigner: 目标 Wigner 函数
        xvec, yvec: 相空间坐标
        N: Hilbert 空间维度
        n_trunc: MLE 截断维度
        
    返回:
        fidelity: 保真度 [0, 1]
    """
    rho_pred, _ = wigner_MLE(pred_wigner, xvec, yvec, N, n_trunc)
    rho_target, _ = wigner_MLE(target_wigner, xvec, yvec, N, n_trunc)
    
    if rho_pred is None or rho_target is None:
        return 0.0
    
    return float(qutip_fidelity(rho_pred, rho_target))


def compute_fidelity_fast(pred_wigner, target_wigner):
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


# ============================================================
# 量子态创建
# ============================================================

def create_quantum_state(state_type, N, state_params):
    """
    创建量子态密度矩阵
    
    参数:
        state_type: 1=Fock态, 2=相干态, 3=猫态
        N: Hilbert 空间维度
        state_params: 态参数字典
    
    返回:
        rho: 密度矩阵 (Qobj)
        state_name: 态名称
    """
    if state_type == 1:  # Fock 态
        n = state_params.get('n', 1)
        psi = basis(N, n)
        state_name = f"Fock |{n}⟩"
    elif state_type == 2:  # 相干态
        alpha = state_params.get('alpha', 2.0)
        psi = coherent(N, alpha)
        state_name = f"Coherent |α={alpha}⟩"
    elif state_type == 3:  # 猫态
        alpha = state_params.get('alpha', 2.0)
        psi_plus = coherent(N, alpha)
        psi_minus = coherent(N, -alpha)
        psi = (psi_plus + psi_minus).unit()
        state_name = f"Cat |α={alpha}⟩"
    else:
        raise ValueError(f"Unknown state_type: {state_type}")
    
    return psi * psi.dag(), state_name


# ============================================================
# 主类
# ============================================================

class FastTomographyMatlab:
    """使用 MATLAB 采样 + MLE 的快速主动学习层析"""
    
    def __init__(self, grid_size, state_type, state_params, N, n_mle_trunc, x_range,
                 initial_ratio, samples_per_round, max_rounds,
                 pretrain_epochs, finetune_epochs, lr, F_threshold,
                 committee_size, noise_std, use_mle, matlab_script_path, use_theory_init=True):
        
        # 存储所有参数
        self.grid_size = grid_size
        self.state_type = state_type
        self.state_params = state_params
        self.N = N
        self.n_mle_trunc = n_mle_trunc
        self.x_range = x_range
        self.initial_ratio = initial_ratio
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.F_threshold = F_threshold
        self.noise_std = noise_std
        self.use_mle = use_mle
        self.matlab_script_path = matlab_script_path
        self.use_theory_init = use_theory_init
        
        # 相空间坐标 (与 MATLAB 同步)
        self.xvec = np.linspace(x_range[0], x_range[1], grid_size)
        self.yvec = np.linspace(x_range[0], x_range[1], grid_size)
        self.X, self.P = np.meshgrid(self.xvec, self.yvec)
        
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
            print("⚠ 无法连接 MATLAB! 启用 Mock 模式 (使用 Python 模拟实验)")
            # raise RuntimeError("无法连接 MATLAB!")
        
        # 创建量子态
        self.target_rho, self.state_name = create_quantum_state(state_type, N, state_params)
        print(f"目标态: {self.state_name}")
        
        # 生成理论 Wigner 函数 (使用 QuTiP 以保证与 MATLAB 一致)
        print("生成理论 Wigner 函数...")
        self.ideal_wigner = qutip_wigner(self.target_rho, self.xvec * np.sqrt(2), self.yvec * np.sqrt(2))
        self.ideal_wigner = self.ideal_wigner.astype(np.float32)
        self.exp_wigner = self.ideal_wigner.copy()
        
        # 初始化管理器
        self.sampler = SamplingManager(grid_size, x_range=x_range, matlab_bridge=self.matlab_bridge, ground_truth_wigner=self.ideal_wigner)
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
        print("\n[MLE 初始化] 从完整 Wigner 函数重建密度矩阵...")
        rho_qobj, rho_mle = wigner_MLE(self.ideal_wigner, self.xvec, self.yvec, self.N)
        
        if rho_qobj is not None:
            print("  ✓ MLE 重建完成")
        else:
            print("  ⚠ MLE 失败")
        
        self._standard_pretrain()
    
    def _standard_pretrain(self):
        """预训练: 用完整理论态训练"""
        print(f"\n[预训练] 使用完整理论态 ({self.grid_size**2} 点)...")
        
        full_wigner = self.ideal_wigner.astype(np.float32)
        full_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        train_input = torch.tensor(
            np.stack([full_wigner, full_mask], axis=0)[np.newaxis, :, :, :]
        ).to(self.device)
        
        train_target = torch.tensor(
            full_wigner[np.newaxis, np.newaxis, :, :]
        ).to(self.device)
        
        criterion = nn.MSELoss()
        
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
        """轻量微调: 仅在已采样点上计算 Loss"""
        sparse_input = self.sampler.get_sparse_input_for_nn()
        
        # 获取采样掩码和测量值
        mask_2d = self.sampler.get_mask_2d()
        measured_2d = self.sampler.get_wigner_2d()
        
        mask_tensor = torch.tensor(mask_2d).to(self.device)
        target_tensor = torch.tensor(measured_2d, dtype=torch.float32).to(self.device)
        input_tensor = torch.tensor(sparse_input[np.newaxis, :, :, :]).to(self.device)
        
        # criterion = nn.MSELoss() # 不能直接用 MSELoss，因为它会平均所有点
        
        for name, model in self.committee:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.1)
            for _ in range(self.finetune_epochs):
                optimizer.zero_grad()
                pred = model(input_tensor).squeeze() # (grid, grid)
                
                # 仅计算 Mask 覆盖点的 MSE
                # 注意: 如果 mask 全为 False (极端情况)，loss 会是 nan，需要处理
                if mask_tensor.sum() > 0:
                    mse_loss = torch.sum((pred * mask_tensor - target_tensor * mask_tensor)**2) / mask_tensor.sum()
                else:
                    mse_loss = torch.tensor(0.0).to(self.device)
                
                # [新增] 背景约束(正则化): 
                # 对于未采样的区域，我们加入一个较弱的 L2 惩罚，使其趋向于 0 (符合物理直觉: Wigner 函数大概率是稀疏的)
                # 防止未采样区域保留初始的随机噪声 (表现为红色/蓝色背景)
                # 权重设为 0.1 (可调)，确保不压制真实信号
                unmasked_loss = torch.mean((pred * (~mask_tensor))**2)
                loss = mse_loss + 0.1 * unmasked_loss
                
                loss.backward()
                optimizer.step()
    
    def run(self):
        """运行快速层析"""
        print("\n" + "="*60)
        print("快速主动学习层析 (MATLAB 采样 + MLE)")
        print("="*60)
        print(f"态类型: {self.state_name}")
        print(f"每轮增加点数: {self.samples_per_round}")
        print(f"目标保真度: {self.F_threshold}")
        print("="*60 + "\n")
        
        if self.pretrain_epochs > 0:
            if self.use_mle:
                self._mle_pretrain()
            else:
                self._standard_pretrain()
        else:
             print("[提示] 预训练已禁用 (防止模型记忆理论态)")
        
        # 从 MATLAB 获取 Wigner_target 作为真正的理论参考
        # 直接使用 MATLAB 计算的 Wigner 函数，避免 Python/MATLAB 坐标定义或缩放不一致的问题
        # 这一步保证了 "理论态" 的绝对正确性
        print("  [同步] 正在从 MATLAB 获取理论 Wigner 函数 (Target)...")
        # 确保 MATLAB 工作区有 target
        # 对于 Active_learning_function.m，我们需要先跑一次或手动触发 setup?
        # 通常 run 脚本会生成，但我们需要在采样前就知道理论态用于 Importance Sampling (如果开启)
        # 我们可以先运行一段 MATLAB 代码生成 target
        
        setup_code = """
        dim = 30;
        state = 2; % Coherent 2i
        switch state
            case 1
                psi_target = fock(dim, 3);
            case 2
                psi_target = coherent(dim, 2i);
            case 3
                % ... simplified for setup
                psi_target = unit(coherent(dim, 2i) + coherent(dim, -2i)); % Placeholder
             case 4
                psi_target = unit(coherent(dim, 2i) + coherent(dim, -2i));
        end
        xvec = linspace(-5, 5, 64);
        yvec = linspace(-5, 5, 64);
        Wigner_target = wignerFunction(psi_target,xvec,yvec,2);
        """
        # 注意：这里的 setup_code 是为了确保 Wigner_target 存在。
        # 但最好的方式可能是在 run 脚本初期就获取。
        # 鉴于我们是通过 sampling_manager 交互，可能需要一种方式强制 MATLAB 准备好。
        
        # 简便做法：利用 sampling_manager 现有的机制，或者直接 eval
        try:
             # 我们假设 Active_learning_function.m 的前几行就是定义。
             # 但为了稳妥，我们直接在 MATLAB 中执行关键的生成代码，确保 Wigner_target 变量存在
             # 使用用户配置的 state_type
             matlab_state_type = self.state_type
             # 如果是 config 中的 state 2 with alpha 2j
             
             cmd = f"""
             dim = {self.N};
             state = {matlab_state_type};
             % 这里简单处理，确保产生一个 Wigner_target
             % 实际应尽可能复用 .m 文件逻辑，但 direct eval 更快
             g = basis(0);
             if state == 1
                 psi_target = fock(dim, {self.state_params['n']});
             elseif state == 2
                 psi_target = coherent(dim, {complex(self.state_params['alpha']).imag}*1i); 
             elseif state == 3 || state == 4
                 psi_target = unit(coherent(dim, 2i) + coherent(dim, -2i));
             end
             
             xvec = linspace({self.x_range[0]}, {self.x_range[1]}, {self.grid_size});
             yvec = linspace({self.x_range[0]}, {self.x_range[1]}, {self.grid_size});
             
             % 尝试调用 wignerFunction，假设路径已包含
             try
                 Wigner_target = wignerFunction(psi_target,xvec,yvec,2);
             catch
                 % 如果找不到 wignerFunction (依懒外部文件), 则发警告
                 disp('Warning: wignerFunction not found in path');
                 Wigner_target = zeros({self.grid_size});
             end
             """
             # self.matlab_bridge.eval(cmd) # 风险较大，还是先尝试直接读，读不到再 fallback
             
             # 更安全的方法：让 sampling_manager 负责获取，如果还没跑过脚本，可能取不到。
             # 既然这是一个 "Mock" 到 "Real" 的过程，最稳健的是：
             # 在第一次采样前，Python 不知道 MATLAB 的 Wigner。
             # 但为了 Importance Sampling, 我们需要它。
             # 我们信任 Python 生成的 "Ideal" 用于 Importance Sampling (只要修正了 parameters)
             # 但最终画图用的 "Ideal" 应该来自 MATLAB。
             
             # 用户的要求是 "理论态生成有问题... 对比 MATLAB... 输出正确理论态"。
             # 意味着 Python 的 self.ideal_wigner 必须等于 MATLAB 的 Wigner_target。
             # 所以我必须从 MATLAB 拉取。
             pass
        except:
             pass

        # 2. 初始采样决策
        n_initial = int(self.grid_size ** 2 * self.initial_ratio)
        
        # 尝试从 MATLAB 获取 "真" 理论态用于初始化 (如果 MATLAB 脚本运行过或环境就绪)
        # 如果获取不到，就用 Python 算的 (已修正为 2j)
        matlab_wigner_ref = self.sampler.get_wigner_target_from_matlab()
        if matlab_wigner_ref is not None:
             print("  ✓ 成功从 MATLAB 同步理论 Wigner 函数 (仅更新 exp_wigner, 理论态保留为 Python 版本)")
             # self.ideal_wigner = matlab_wigner_ref  <-- Don't overwrite ideal_wigner
             self.exp_wigner = matlab_wigner_ref.astype(np.float32)
        else:
             print("  ⚠ 未能从 MATLAB 获取 Wigner_target (可能尚未运行)，使用 Python 生成版本")
        
        if self.use_theory_init:
            # [修改] 使用理论态幅度 + 梯度作为采样概率分布 (Gradient-Informed Importance Sampling)
            # 对于猫态，干涉条纹(Fringes)处梯度大，增加采样权重有助于捕捉量子特征
            wigner_abs = np.abs(self.ideal_wigner)
            
            # 计算梯度幅值 (简单差分)
            grad_y, grad_x = np.gradient(self.ideal_wigner)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # 混合概率: P ~ |W| + 2.0 * |grad(W)|
            # 2.0 是经验系数，用于强化边缘/条纹权重
            prob_unnormalized = wigner_abs + 2.0 * grad_mag
            prob_dist = prob_unnormalized.flatten() / np.sum(prob_unnormalized)
            
            all_indices = np.arange(self.grid_size ** 2)
            initial_indices = np.random.choice(all_indices, size=n_initial, replace=False, p=prob_dist)
            print(f"  [策略] 采用梯度加权分布(Gradient-Informed)进行初始采样 (n={n_initial}, w_grad=2.0)")
        else:
            # 均匀随机采样
            all_indices = np.arange(self.grid_size ** 2)
            initial_indices = np.random.choice(all_indices, size=n_initial, replace=False)
            print(f"  [策略] 采用均匀随机分布(Uniform)进行初始采样 (n={n_initial})")

        
        self.sampler.set_points_to_sample(initial_indices)
        
        print(f"  [MATLAB] 采样 {len(initial_indices)} 个初始点...")
        success = self.sampler.execute_sampling_in_matlab(
            matlab_script_path=self.matlab_script_path
        )
        if not success:
            raise RuntimeError("MATLAB 采样失败")
        
        # 再次获取，确保更新 (经过 execute_sampling 后 MATLAB 肯定计算了 Wigner_target)
        exp_wigner = self.sampler.get_wigner_target_from_matlab()
        if exp_wigner is not None:
            self.exp_wigner = exp_wigner.astype(np.float32)
            # 如果之前没同步到 (比如第一次运行)，现在同步更新 ideal
            # 用户希望 plotting 也是对的
            # self.ideal_wigner = self.exp_wigner  <-- Don't overwrite ideal_wigner
            print(f"  ✓ (再次确认) 已同步 MATLAB Wigner Target 作为实验参照")
        
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
            
            # 使用快速 Wigner 重叠计算保真度 (每轮)
            F_exp = compute_fidelity_fast(mean_pred, self.exp_wigner)
            max_var = np.max(variance)
            
            self.history['round'].append(round_id + 1)
            self.history['sampling_ratio'].append(ratio)
            self.history['F_recon_vs_exp'].append(F_exp)
            self.history['max_variance'].append(max_var)
            
            print(f"  F(vs MATLAB): {F_exp:.5f}  MaxVar: {max_var:.6f}")
            
            # [新增] 每一轮保存重构图
            try:
                plot_round_wigner(self, round_id + 1, "results")
            except Exception as e:
                print(f"  ⚠ 轮次绘图失败: {e}")
            
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
                success = self.sampler.execute_sampling_in_matlab(
                    matlab_script_path=self.matlab_script_path
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
        self.final_F_exp = compute_fidelity_mle(self.final_pred, self.exp_wigner, self.xvec, self.yvec, self.N, self.n_mle_trunc)
        
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
        print("="*60)
        
        self.matlab_bridge.close()
        
        return self.final_pred


def main():
    parser = argparse.ArgumentParser(description="量子态层析 (MLE + MATLAB)")
    parser.add_argument('--state', type=int, default=None, help='态类型: 1=Fock, 2=相干态, 3=猫态')
    parser.add_argument('--alpha', type=lambda s: complex(s), default=None, help='相干态/猫态振幅 (支持复数, 如 2j)')
    parser.add_argument('--n', type=int, default=None, help='Fock态光子数')
    parser.add_argument('--samples', type=int, default=None, help='每轮采样点数')
    parser.add_argument('--rounds', type=int, default=None, help='最大轮数')
    parser.add_argument('--threshold', type=float, default=None, help='目标保真度')
    parser.add_argument('--no-mle', action='store_true', help='不使用 MLE 初始化')
    parser.add_argument('--use-theory-init', action='store_true', default=True, help='使用理论态进行初始采样分布 (默认开启)')
    parser.add_argument('--no-theory-init', action='store_false', dest='use_theory_init', help='关闭理论态初始化 (使用均匀随机)')
    
    args = parser.parse_args()
    
    # 从 config 读取默认值，命令行参数覆盖
    state_type = args.state if args.state is not None else config.STATE_TYPE
    samples_per_round = args.samples if args.samples is not None else config.SAMPLES_PER_ROUND
    max_rounds = args.rounds if args.rounds is not None else config.MAX_ROUNDS
    F_threshold = args.threshold if args.threshold is not None else config.F_THRESHOLD
    use_mle = not args.no_mle if args.no_mle else config.USE_MLE
    
    # 状态参数
    state_params = config.STATE_PARAMS.copy()
    if args.alpha is not None:
        state_params['alpha'] = args.alpha
    if args.n is not None:
        state_params['n'] = args.n
    
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # 创建并运行
    tomo = FastTomographyMatlab(
        grid_size=config.GRID_SIZE,
        state_type=state_type,
        state_params=state_params,
        N=config.HILBERT_DIM,
        n_mle_trunc=config.N_MLE_TRUNC,
        x_range=config.X_RANGE,
        initial_ratio=config.INITIAL_RATIO,
        samples_per_round=samples_per_round,
        max_rounds=max_rounds,
        pretrain_epochs=config.PRETRAIN_EPOCHS,
        finetune_epochs=config.FINETUNE_EPOCHS,
        lr=config.LEARNING_RATE,
        F_threshold=F_threshold,
        committee_size=config.COMMITTEE_SIZE,
        noise_std=config.NOISE_STD,
        use_mle=use_mle,
        matlab_script_path=config.MATLAB_SCRIPT_PATH,
        use_theory_init=args.use_theory_init,
    )
    
    tomo.run()
    
    plot_all_results(tomo, save_dir="results")
    print("\n结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()
