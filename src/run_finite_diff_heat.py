from method_finite_difference import forward_difference, central_difference, second_derivative
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_1d(dx=0.1, dt=0.01, T=1.0, L=1.0, alpha=1.0):
    """Solve the 1D heat equation using explicit finite difference method.
    Returns temperature distribution over space and time, an np.ndarray.
    Can plot heat map.

    Parameters:
    dx (float): Space step
    dt (float): Time step
    T (float): Total simulation time
    L (float): Length of the domain
    alpha (float): Diffusion coefficient
    plot (bool): Create plot true or false
    """

    nx = int(L / dx) + 1
    nt = int(T / dt) + 1

    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError("Time step too large, unstable: reduce dt.")
    
    u = np.zeros((nt, nx))
    x = np.linspace(0, L, nx)
    u[0, :] = np.sin(np.pi * x)
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    return x, u

f = lambda x: np.sin(x)
x0 = np.pi / 4
print(f"Forward Difference: {forward_difference(f, x0)}")
print(f"Central Difference: {central_difference(f, x0)}")
print(f"Second Derivative: {second_derivative(f, x0)}")

x, u = solve_heat_1d(dx=0.05, dt=0.001, T=0.5, L=1.0, alpha=1.0, plot=True)
plt.imshow(u, aspect='auto', cmap='hot', origin='lower', extent=[0, 1, 0, 1])
plt.title("Heat Equation Solution")
plt.colorbar(label="Temperature")
plt.xlabel("Position")
plt.ylabel("Time")
plt.show()