
import numpy as np
from qutip import coherent, wigner, Qobj
import matplotlib.pyplot as plt

# User's parameters
alpha = 2j
N = 50
grid_size = 64
x_range = (-5, 5)

xvec = np.linspace(x_range[0], x_range[1], grid_size)
yvec = np.linspace(x_range[0], x_range[1], grid_size)

rho = coherent(N, alpha)
rho = rho * rho.dag()

# Current generation method in main.py
W = wigner(rho, xvec, yvec)

# Find peak
max_idx = np.unravel_index(np.argmax(W), W.shape)
peak_x = xvec[max_idx[0]] # QuTiP wigner returns W[x, y] or W[y, x]? Usually W[x_idx, y_idx] corresponds to xvec[x_idx], yvec[y_idx]? 
# QuTiP wigner returns array of shape (len(xvec), len(yvec)).
# Let's verify coordinates.
peak_val = np.max(W)
# x is usually row or col?
# In qutip wigner(rho, x, y):
# rows correspond to x, columns to y? Or vice versa?
# Typically it matches meshgrid.

print(f"Peak value: {peak_val}")
print(f"Peak indices: {max_idx}")
# xvec is 1st dim, yvec is 2nd dim?
# Actually qutip wigner typically does: x (rows), p (cols) or similar.
# Let's check coordinates.

# If we assume standard QuTiP:
# W[i, j] corresponds to xvec[i] and yvec[j].
peak_x_coord = xvec[max_idx[0]]
peak_y_coord = yvec[max_idx[1]]

print(f"Peak location in (xvec, yvec): ({peak_x_coord:.3f}, {peak_y_coord:.3f})")

# Expected for alpha=2j if xvec,yvec are quadratures:
# Re(alpha)=0, Im(alpha)=2.
# x = sqrt(2)*Re = 0
# p = sqrt(2)*Im = 2.828
print(f"Expected peak for Quadratures: (0, {2*np.sqrt(2):.3f})")
print(f"Expected peak for Alpha plane: (0, 2.000)")

