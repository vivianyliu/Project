from method_lax_wendroff import solve_laxwendroff_linear, compute_phase_error
import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 200
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
u0 = np.sin(2 * np.pi * x) # Initial condition: we choose single sine wave
a = 1.0
dt = 0.4 * dx / a
t_max = 1.0
u_exact = np.sin(2 * np.pi * ((x - a * t_max) % 1.0))
u_hist = solve_laxwendroff_linear(u0, a, dx, dt, t_max)
plt.plot(x, u_exact, '--', label='Exact')
plt.plot(x, u_hist[-1], label="Lax-Wendroff")
plt.title(f"Lax-Wendroff Basic Solution at t={t_max}")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()

shift = compute_phase_error(u_hist, u_exact)
print(f"Approximate phase shift (in grid points): {shift}")