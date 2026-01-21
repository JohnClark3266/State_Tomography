import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import os

# Ensure output directory exists
os.makedirs('ppt/images', exist_ok=True)

def wigner_fock(x, p, n):
    """Ref: W_n(x,p) = (-1)^n/pi * exp(-(x^2+p^2)) * L_n(2(x^2+p^2))"""
    r2 = x**2 + p**2
    return ((-1)**n / np.pi) * np.exp(-r2) * genlaguerre(n, 0)(2 * r2)

def wigner_coherent(x, p, alpha_real, alpha_imag):
    """Gaussian centered at sqrt(2)*alpha"""
    # W(alpha) = 2/pi * exp(-2|beta-alpha|^2)
    # x, p are related to beta by beta = (x + ip)/sqrt(2)
    # alpha is the coherent state parameter
    
    # We use the convention where vacuum variance is 1/2? 
    # Or common Wigner convention where vacuum has width 1.
    # Let's align with the previous code: domain [-4, 4]
    
    # Standard Wigner for coherent state |alpha>
    # W(x, p) = (1/pi) * exp( -((x - x0)^2 + (p - p0)^2) )
    # where x0 = sqrt(2)*Re(alpha), p0 = sqrt(2)*Im(alpha)
    
    x0 = np.sqrt(2) * alpha_real
    p0 = np.sqrt(2) * alpha_imag
    return (1.0 / np.pi) * np.exp(-((x - x0)**2 + (p - p0)**2))

def wigner_cat(x, p, alpha=2.0):
    """Schrödinger Cat State |psi> ~ |alpha> + |-alpha>"""
    # Unnormalized approximation for visualization
    # W_cat = W_alpha + W_minus_alpha + Interference
    
    # Interference term:
    # W_int ~ cos(2*p*x0 - 2*x*p0) * exp(...)
    
    # Exact form for N(|alpha> + |-alpha>)
    # W(x,p) ~ e^{-(x-x0)^2 - (p-p0)^2} + e^{-(x+x0)^2 - (p+p0)^2} 
    #          + 2 * e^{-x^2-p^2} * cos(2*x0*p - 2*p0*x)
    
    x0 = np.sqrt(2) * alpha
    # Assume alpha is real for simplicity -> p0 = 0
    t1 = np.exp(-((x - x0)**2 + p**2))
    t2 = np.exp(-((x + x0)**2 + p**2))
    interference = 2 * np.exp(-(x**2 + p**2)) * np.cos(2 * x0 * p)
    
    return (1.0 / (2 * np.pi * (1 + np.exp(-2*alpha**2)))) * (t1 + t2 + interference)

def plot_wigner(func, title, filename, **kwargs):
    x = np.linspace(-4, 4, 200)
    p = np.linspace(-4, 4, 200)
    X, P = np.meshgrid(x, p)
    W = func(X, P, **kwargs)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(X, P, W, levels=100, cmap='RdBu_r')
    plt.colorbar(label='W(x, p)')
    plt.title(title)
    plt.xlabel('Position x')
    plt.ylabel('Momentum p')
    plt.tight_layout()
    plt.savefig(f'ppt/images/{filename}', dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # 1. Fock State |3>
    plot_wigner(wigner_fock, r'Fock State $|3\rangle$ Wigner Function', 'wigner_fock3.png', n=3)
    
    # 2. Coherent State |alpha=2>
    plot_wigner(wigner_coherent, r'Coherent State $|\alpha=2\rangle$', 'wigner_coherent.png', alpha_real=2.0, alpha_imag=0.0)
    
    # 3. Cat State
    plot_wigner(wigner_cat, r'Schrödinger Cat State', 'wigner_cat.png', alpha=2.0)
