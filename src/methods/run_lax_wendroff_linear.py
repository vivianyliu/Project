from lax_wendroff import lax_wendroff_linear, plot_solution_comparison, compute_phase_error
import numpy as np

L = 1.0
N = 200
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
# Initial condition: we choose single sine wave
u0 = np.sin(2 * np.pi * x)
a = 1.0
dt = 0.4 * dx / a
t_max = 1.0
u_exact = np.sin(2 * np.pi * ((x - a * t_max) % 1.0))
u_hist = lax_wendroff_linear(u0, a, dx, dt, t_max)
plot_solution_comparison(x, u_hist, u_exact, t_max, label="Lax-Wendroff")
shift = compute_phase_error(u_hist, u_exact)
print(f"Approximate phase shift (in grid points): {shift}")