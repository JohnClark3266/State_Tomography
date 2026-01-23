"""
Wigner函数生成和MLE密度矩阵重建模块

基于用户提供的 Wigner_MLE_fidelity.py，使用QuTiP进行精确计算。

功能:
- generate_wigner_data: 从密度矩阵生成Wigner函数
- wigner_MLE: 从Wigner测量数据重建密度矩阵
- compute_fidelity_qutip: 使用QuTiP计算保真度
"""

import numpy as np
import cvxpy as cp
from qutip import destroy, displace, basis, Qobj, fidelity as qutip_fidelity, wigner


def parity_op(N):
    """创建宇称算符"""
    return Qobj(np.diag([(-1)**n for n in range(N)]))


def wigner_at_point(rho, beta, parity, N):
    """计算单点Wigner值"""
    D = displace(N, beta)
    return (D * parity * D.dag() * rho).tr().real


def generate_wigner_data(rho, xvec, yvec, N):
    """
    从密度矩阵生成Wigner函数数据
    
    参数:
        rho: QuTiP Qobj, 密度矩阵
        xvec: array, x轴坐标
        yvec: array, y轴坐标
        N: int, Hilbert空间截断维度
    
    返回:
        W_data: ndarray, Wigner函数值
        parity: Qobj, 宇称算符
    """
    parity = parity_op(N)
    W_data = np.zeros((len(xvec), len(yvec)))
    for ix, x in enumerate(xvec):
        for iy, y in enumerate(yvec):
            beta = x + 1j*y
            W_data[ix, iy] = wigner_at_point(rho, beta, parity, N)
    return W_data, parity


def wigner_MLE(MeasWigner, xvec, yvec, N, n_trunc, verbose=False):
    """
    最大似然估计从Wigner测量数据重建密度矩阵
    
    参数:
        MeasWigner: ndarray, 测量的Wigner函数 (可以是稀疏的)
        xvec: array, x轴坐标
        yvec: array, y轴坐标
        N: int, Hilbert空间总维度
        n_trunc: int, 重建密度矩阵的维度
        verbose: bool, 是否显示求解器输出
    
    返回:
        rho_qobj: Qobj, 重建的密度矩阵
        rho_mle: ndarray, 重建的密度矩阵数组
    """
    # Flatten Wigner data
    W_vec = MeasWigner.T.reshape(-1)
    
    # Build design matrix
    beta_list = [x + 1j*y for y in yvec for x in xvec]
    num_meas = len(beta_list)
    parity = parity_op(N)
    A_real = np.zeros((num_meas, n_trunc**2))
    
    for m, beta in enumerate(beta_list):
        D = displace(N, beta)
        M = D * parity * D.dag()
        M = M.full()[:n_trunc, :n_trunc]
        A_real[m, :] = M.real.T.reshape(-1)
    
    # MLE via cvxpy (real optimization)
    rho_var = cp.Variable((n_trunc, n_trunc), symmetric=True)
    constraints = [rho_var >> 0, cp.trace(rho_var) == 1]
    objective = cp.Minimize(cp.norm(A_real @ cp.vec(rho_var) - W_vec, 2))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)
    
    rho_mle = rho_var.value
    rho_qobj = Qobj(rho_mle)
    return rho_qobj, rho_mle


def compute_fidelity_qutip(rho1, rho2):
    """
    使用QuTiP计算两个密度矩阵之间的保真度
    
    参数:
        rho1, rho2: Qobj, 密度矩阵
    
    返回:
        float: 保真度值
    """
    return qutip_fidelity(rho1, rho2)


def wigner_from_qutip(rho, xvec, yvec):
    """
    使用QuTiP内置函数计算Wigner函数
    
    参数:
        rho: Qobj, 密度矩阵
        xvec: array, x轴坐标
        yvec: array, y轴坐标
    
    返回:
        ndarray: Wigner函数值
    """
    return wigner(rho, xvec, yvec)


# 便捷函数：创建常用量子态
def create_fock_state(N, n):
    """创建Fock态 |n⟩"""
    return basis(N, n) * basis(N, n).dag()


def create_coherent_state(N, alpha):
    """创建相干态 |α⟩"""
    a = destroy(N)
    D = displace(N, alpha)
    return D * basis(N, 0) * basis(N, 0).dag() * D.dag()


def create_cat_state(N, alpha, parity='even'):
    """创建猫态"""
    D_plus = displace(N, alpha)
    D_minus = displace(N, -alpha)
    vac = basis(N, 0)
    if parity == 'even':
        psi = (D_plus * vac + D_minus * vac).unit()
    else:
        psi = (D_plus * vac - D_minus * vac).unit()
    return psi * psi.dag()
