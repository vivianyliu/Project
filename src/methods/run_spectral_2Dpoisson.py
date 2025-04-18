from spectral_methods import solve_poisson_spectral_2D, compute_error_up_to_constant
import matplotlib.pyplot as plt
import numpy as np

f = lambda x, y: np.exp(np.sin(x) + np.cos(y))
X, Y, u = solve_poisson_spectral_2D(f)
f_vals = f(X, Y)
u_exact = -f_vals
error = compute_error_up_to_constant(u, u_exact)
plt.figure(figsize=(7, 5))
cp = plt.contourf(X, Y, u, 50, cmap='viridis')
plt.colorbar(cp, label='u(x, y)')
plt.title(r"Solution of $\nabla^2 u = -e^{\sin(x)+\cos(y)}$ via Spectral Method")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(1, 1, f'error = {error:.2e}', fontsize=9, bbox=props)
plt.show()