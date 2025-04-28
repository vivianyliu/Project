from method_finite_difference import forward_difference, central_difference, second_derivative
import numpy as np
import matplotlib.pyplot as plt

def solve_wave_1d(dx=0.1, dt=0.01, T=1.0, L=1.0, c=1.0):
    """Solve the 1D wave equation using finite difference method.
    Returns wave distribution over space and time, an np.ndarray.
    Can plot wave propogation.

    Parameters:
    dx (float): Space step
    dt (float): Time step
    T (float): Total simulation time
    L (float): Length of the domain
    c (float): Wave speed
    plot (bool): Create plot true or false
    """

    nx = int(L / dx) + 1
    nt = int(T / dt) + 1

    if dt > dx / c:
        raise ValueError("Time step too large, unstable: reduce dt or increase dx.")

    u = np.zeros((nt, nx))
    x = np.linspace(0, L, nx)
    u[0, :] = np.exp(-100 * (x - L/2)**2)
    u[:, 1] = u[:, 0]
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = (2 * u[n, i] - u[n-1, i] + 
                         (c**2 * dt**2 / dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1]))
    return x, u, nt, dt

f = lambda x: np.sin(x)
x0 = np.pi / 4
print(f"Forward Difference: {forward_difference(f, x0)}")
print(f"Central Difference: {central_difference(f, x0)}")
print(f"Second Derivative: {second_derivative(f, x0)}")

x, u, nt, dt = solve_wave_1d(dx=0.05, dt=0.001, T=1.0, L=1.0, c=1.0)
# x, u, nt, dt = solve_wave_1d()
for n in range(0, nt, nt//10):
    plt.plot(x, u[n, :], label=f"t={n*dt:.2f}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("1D Wave Equation - Finite Differences")
plt.legend()
plt.grid()
plt.show()