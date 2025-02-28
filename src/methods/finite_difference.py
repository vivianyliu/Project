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

def dDifference(x,y):
    #divided differences that takes in 2 arrays in the case of unequal step sizes
    n = x.size
    y1 = y.copy()
    dd = np.zeros(n)
    for i in range(n):
        dd[i] = y1[i]
    for j in range(1,n):
        for i in range(n-j):
            y1[i] = (y1[i + 1] - y1[i]) / (x[i + j] - x[i])
        dd[j] = y1[0]  
    return (dd)

def interpPoly(xx,yy):
    #interpolation function using divided diff 
    dd = dDifference(xx,yy)
    def poly(x):
        n = xx. size
        p = dd[n-1]
        for i in range(n-2,-1,-1):
            p = p * (x-xx[i]) + dd[i]
        return (p)
    return (poly)

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

def wave_equation_1d(L=5.0, Nx=50, c=2.0, dt=0.02, Nt=50, plot=True):
    dx = L / Nx  # Spatial step
    assert dt <= dx / c, "Time step too large, decrease dt or increase dx for stability"

    # Initialize the wave field
    u = np.zeros((Nx, Nt))  # Solution grid (space x time)

    # Define spatial grid and set initial condition (Gaussian pulse)
    x = np.linspace(0, L, Nx)
    u[:, 0] = np.exp(-100 * (x - L/2)**2)  # Initial displacement
    u[:, 1] = u[:, 0]  # Assume stationary start (zero initial velocity)

    # Time stepping loop using finite difference method
    for n in range(1, Nt-1):
        for i in range(1, Nx-1):  # Avoid boundary points
            u[i, n+1] = (2 * u[i, n] - u[i, n-1] +
                         (c**2 * dt**2 / dx**2) * (u[i+1, n] - 2*u[i, n] + u[i-1, n]))

    # Plot wave propagation
    if plot:
        plt.figure(figsize=(8, 5))
        for n in range(0, Nt, Nt//10):  # Plot snapshots at different times
            plt.plot(x, u[:, n], label=f"t={n*dt:.2f}")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("1D Wave Equation - Finite Differences")
        plt.legend()
        plt.grid()
        plt.show()

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
    wave_equation_1d()