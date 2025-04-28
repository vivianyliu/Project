from method_lax_wendroff import solve_laxwendroff_burgers
import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 200
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
u0 = np.sin(2 * np.pi * x)
dt = 0.3 * dx / np.max(np.abs(u0)) # conservative CFL
t_max = 0.5
u_hist = solve_laxwendroff_burgers(u0, dx, dt, t_max)
plt.plot(x, u0, '--', label="Initial")
plt.plot(x, u_hist[-1], label="Lax-Wendroff (Burgers)")
plt.title("Nonlinear Advection with Lax-Wendroff: Inviscid Burgers' Equation")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()