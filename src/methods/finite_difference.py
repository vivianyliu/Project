import numpy as np
import matplotlib.pyplot as plt

def forward_difference(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-6):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-6):
    #more accurate than forwards or backwards
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    #f''(x) using the second-order central difference method.
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def solve_heat_equation(dx=0.1, dt=0.01, T=1.0, L=1.0, alpha=1.0):
    #alpha (float): Diffusion coefficient
    
    nx = int(L / dx) + 1  # Number of spatial points
    nt = int(T / dt) + 1  # Number of time steps
    u = np.zeros((nt, nx))  # Solution array
    
    # Initial condition u(x,0) = sin(pi*x)
    x = np.linspace(0, L, nx)
    u[0, :] = np.sin(np.pi * x)

    # Stability criteria for explicit scheme
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError("Time step too large, solution may be unstable. Reduce dt.")

    # Time stepping loop
    for n in range(0, nt - 1):
        for i in range(1, nx - 1): # Interior points only
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

    return x, u

if __name__ == "__main__": # test main for now
    f = lambda x: np.sin(x)
    x0 = np.pi / 4
    print(f"Forward Difference: {forward_difference(f, x0)}")
    print(f"Central Difference: {central_difference(f, x0)}")
    print(f"Second Derivative: {second_derivative(f, x0)}")

    x1, u1 = solve_heat_equation(dx=0.05, dt=0.001, T=0.5, L=1.0, alpha=1.0)
    plt.imshow(u1, aspect='auto', cmap='hot', origin='lower', extent=[0, 1, 0, 1])
    plt.title("Heat Equation Solution")
    plt.colorbar(label="Temperature")
    plt.xlabel("Position")
    plt.ylabel("Time")
    plt.show()