from method_lax_wendroff import solve_laxwendroff_burgers
import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 200
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
u0_burgers = np.sin(2 * np.pi * x)
dt_burgers = 0.3 * dx / np.max(np.abs(u0_burgers)) # conservative CFL
t_max_burgers = 0.5
u_hist_burgers = solve_laxwendroff_burgers(u0_burgers, dx, dt_burgers, t_max_burgers)
plt.plot(x, u0_burgers, '--', label="Initial")
plt.plot(x, u_hist_burgers[-1], label="Lax-Wendroff (Burgers)")
plt.title("Nonlinear Advection: Inviscid Burgers' Equation")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()