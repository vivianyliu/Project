from method_spectral_methods import solve_poisson_spectral_1D, compute_error_up_to_constant
import matplotlib.pyplot as plt
import numpy as np

f = lambda x: np.exp(np.sin(x))
x, u = solve_poisson_spectral_1D(f)
err = compute_error_up_to_constant(u, -f(x))
plt.plot(x, u, label='Spectral Solution')
plt.plot(x, -f(x), '--', label='-f(x) =-e^(sin(x))', alpha=0.5)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title("1D Spectral Poisson Solution of $u''(x) = -e^{sin(x)}$")
plt.legend()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.3, -0.5, f'error ={err}', fontsize=8, bbox = props)
plt.grid(True)
plt.show()