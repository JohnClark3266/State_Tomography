import numpy as np
import cvxpy as cp
from qutip import destroy, displace, basis, Qobj, fidelity, wigner
import matplotlib.pyplot as plt

# Parameters
N = 50           # Hilbert space truncation
n_trunc = N      # MLE reconstruction dimension
xvec = np.linspace(-3, 3, 51)  # Phase-space grid
yvec = np.linspace(-3, 3, 51)

# Wigner generation functions
def parity_op(N):
    return Qobj(np.diag([(-1)**n for n in range(N)]))

def wigner_at_point(rho, beta, parity):
    D = displace(N, beta)
    return (D * parity * D.dag() * rho).tr().real

def generate_wigner_data(rho, xvec, yvec, N):
    parity = parity_op(N)
    W_data = np.zeros((len(xvec), len(yvec)))
    for ix, x in enumerate(xvec):
        for iy, y in enumerate(yvec):
            beta = x + 1j*y
            W_data[ix, iy] = wigner_at_point(rho, beta, parity)
    return W_data, parity

# MLE reconstruction function
def wigner_MLE(MeasWigner, xvec, yvec, N, n_trunc, verbose=False):
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

# Test several states and plot Wigner functions
test_states = {
    "|1>": basis(N, 1),
    "|2>": basis(N, 2),
    "(|0> + |1>)/sqrt(2)": (basis(N,0)+basis(N,1)).unit(),
    "(|1> + |2>)/sqrt(2)": (basis(N,1)+basis(N,2)).unit(),
}

for name, psi in test_states.items():
    print(f"\n--- Testing state: {name} ---")
    rho = psi * psi.dag()
    W_data, _ = generate_wigner_data(rho, xvec, yvec, N)
    
    # MLE reconstruction
    rho_qobj, rho_mle = wigner_MLE(W_data, xvec, yvec, N, n_trunc, verbose=False)
    
    # Fidelity
    fid_val = fidelity(rho, rho_qobj)
    print(f"Fidelity: {fid_val:.4f}")
    
    # Visualize real part of the reconstructed density matrix
    plt.figure(figsize=(5,4))
    plt.imshow(np.real(rho_mle), origin='lower', cmap='viridis')
    plt.colorbar(label='Re[rho]')
    plt.title(f'{name} reconstructed rho (MLE)')
    plt.show()
    
    # Plot Wigner functions: ideal vs reconstructed
    W_ideal = wigner(rho, xvec, yvec)
    W_recon = wigner(rho_qobj, xvec, yvec)
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(W_ideal.T, extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='Wigner')
    plt.title(f'{name} Ideal Wigner')
    
    plt.subplot(1,2,2)
    plt.imshow(W_recon.T, extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='Wigner')
    plt.title(f'{name} Reconstructed Wigner (MLE)')
    
    plt.show()
