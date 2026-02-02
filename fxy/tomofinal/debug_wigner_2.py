
import numpy as np
from qutip import coherent, wigner, Qobj
import matplotlib.pyplot as plt

alpha = 2j
N = 50
grid_size = 64
x_range = (-5, 5)

xvec = np.linspace(x_range[0], x_range[1], grid_size)
yvec = np.linspace(x_range[0], x_range[1], grid_size)
X, P = np.meshgrid(xvec, yvec) # Default 'xy': X varies cols, P varies rows.
# P[i, :] is constant p_vec[i]. 
# So P meshgrid corresponds to yvec being the ROW axis.

rho = coherent(N, alpha)
rho = rho * rho.dag()

print("--- Test 1: Standard wigner(rho, xvec, yvec) ---")
W1 = wigner(rho, xvec, yvec) 
# Usually W[i, j] matches xvec[i], yvec[j].
# So W rows -> xvec, W cols -> yvec.
# But meshgrid P rows -> yvec.
# So W rows -> X ?? No. 
# If W[i, j] is (x[i], y[j]).
# Contourf(X, P, Z) expects Z[row, col] to match (X[row,col], P[row,col]).
# X[row,col] -> x_val. P[row,col] -> y_val.
# If meshgrid is 'xy': X[i,j] = xvec[j], P[i,j] = yvec[i].
# So Z[i,j] should be W(xvec[j], yvec[i]).
# Note indices are swapped!
# So we need Z = W.T? 
# W.T[i,j] = W[j, i] = W(xvec[j], yvec[i]). Correct.
# So we should use W.T for plotting with standard meshgrid.

# Let's check where the peak is in W1.
max_idx1 = np.unravel_index(np.argmax(W1), W1.shape)
print(f"Peak index in W1 (x_idx, y_idx): {max_idx1}")
print(f"Coordinates: x={xvec[max_idx1[0]]:.2f}, y={yvec[max_idx1[1]]:.2f}")
# Previously we saw x=2.78, y=-0.08. 
# This means W is high at (x=2.8, y=0). 
# But theory <p>=2.8. So we suspect W1 actually IS high at y=2.8??
# Or Qutip 5.x vs 4.x differences?
# Assuming modern Qutip.
# Let's verify scaling.

print("\n--- Test 2: Scaled wigner(rho, xvec*sqrt(2), yvec*sqrt(2)) ---")
xvec_scaled = xvec * np.sqrt(2)
yvec_scaled = yvec * np.sqrt(2)
W2 = wigner(rho, xvec_scaled, yvec_scaled)

max_idx2 = np.unravel_index(np.argmax(W2), W2.shape)
print(f"Peak index in W2 (x_idx, y_idx): {max_idx2}")
print(f"Coordinates unscaled: x={xvec[max_idx2[0]]:.2f}, y={yvec[max_idx2[1]]:.2f}")
# If correct, we want peak at alpha_im=2 (y=2).
# So we want y_idx to correspond to 2.0.
# yvec is -5..5. 2.0 is around index 44.
# x_idx should be around 0 (index 31).
# So we want peak at (31, 44) (if W[x_idx, y_idx]).
# Let's see.

