from method_lax_wendroff import solve_laxwendroff_linear, compute_phase_error
import numpy as np
import matplotlib.pyplot as plt

def plot_solution_comparison(x, u_hist, u_exact, t, label="Numerical"):
    """
    Plots the numerical and exact solutions at a given time.

    Parameters:
        x: array-like, spatial grid points
        u_hist: 2D array, numerical solution at all time steps
        u_exact: array-like, exact solution at final time
        t: float, time of snapshot
        label: str, label for the numerical method used
    """
    plt.plot(x, u_exact, '--', label='Exact')
    plt.plot(x, u_hist[-1], label=label)
    plt.title(f"Solution at t={t}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()

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
u_hist = solve_laxwendroff_linear(u0, a, dx, dt, t_max)
plot_solution_comparison(x, u_hist, u_exact, t_max, label="Lax-Wendroff")
shift = compute_phase_error(u_hist, u_exact)
print(f"Approximate phase shift (in grid points): {shift}")