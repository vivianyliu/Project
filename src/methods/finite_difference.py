import numpy as np
import matplotlib.pyplot as plt

def forward_difference(f, x, h=1e-6):
    """To find approximation of f'(x) by forward difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """

    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-6):
    """To find approximation of f'(x) by backward difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """

    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-6):
    """To compute f'(x) using the central difference method.
    More accurate than forward or backward difference.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """
    
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    """Compute f''(x) using the second-order central difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-5)
    """
    
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def div_difference(x, y):
    """Compute divided differences where arrays have unequal step sizes.

    Parameters:
    x (array): First array
    y (array): Second array
    """

    n = x.size
    y1 = y.copy()
    dd = np.zeros(n)

    for i in range(n):
        dd[i] = y1[i]
    for j in range(1, n):
        for i in range(n - j):
            y1[i] = (y1[i + 1] - y1[i]) / (x[i + j] - x[i])
        dd[j] = y1[0]
    
    return (dd)

def interp_poly(xx, yy):
    """Returns the interpolation function when using divided differences.
 
    Parameters:
    x (array): First array
    y (array): Second array
    """

    dd = div_difference(xx, yy)
    def poly(x):
        n = xx.size
        p = dd[n-1]
        for i in range(n-2, -1, -1):
            p = p * (x-xx[i]) + dd[i]
        return (p)
    return (poly)


def solve_heat_equation_1d(dx=0.1, dt=0.01, T=1.0, L=1.0, alpha=1.0, plot=True):
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

    # Number of spatial points
    nx = int(L / dx) + 1

    # Number of time steps
    nt = int(T / dt) + 1

    # Stability criteria for explicit scheme
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError("Time step too large, unstable: reduce dt.")
    
    # Solution grid
    u = np.zeros((nt, nx))
    
    # Initial condition u(x, 0) = sin(pi*x)
    x = np.linspace(0, L, nx)
    u[0, :] = np.sin(np.pi * x)

    # Time stepping loop by explicit finite difference method
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

    if plot:        
        plt.imshow(u, aspect='auto', cmap='hot', origin='lower', extent=[0, 1, 0, 1])
        plt.title("Heat Equation Solution")
        plt.colorbar(label="Temperature")
        plt.xlabel("Position")
        plt.ylabel("Time")
        plt.show()
    
    return x, u

def solve_wave_equation_1d(dx=0.1, dt=0.01, T=1.0, L=1.0, c=1.0, plot=True):
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

    nx = int(L / dx) + 1 # spatial points
    nt = int(T / dt) + 1 # time steps
    
    # Stability condition for wave equation
    if dt > dx / c:
        raise ValueError("Time step too large, unstable: reduce dt or increase dx.")

    # Solution grid wave field, space x time
    u = np.zeros((nt, nx))

    # Define spatial grid, set initial condition (Gaussian pulse)
    x = np.linspace(0, L, nx)
    u[0, :] = np.exp(-100 * (x - L/2)**2)

    # Assume stationary start (zero initial velocity)
    u[:, 1] = u[:, 0]

    # Time stepping loop by finite difference method
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = (2 * u[n, i] - u[n-1, i] + 
                         (c**2 * dt**2 / dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1]))

    if plot:
        plt.figure(figsize=(8, 5))
        for n in range(0, nt, nt//10):
            plt.plot(x, u[n, :], label=f"t={n*dt:.2f}")
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("1D Wave Equation - Finite Differences")
        plt.legend()
        plt.grid()
        plt.show()

    return x, u


if __name__ == "__main__":
    
    f = lambda x: np.sin(x)
    x0 = np.pi / 4
    print(f"Forward Difference: {forward_difference(f, x0)}")
    print(f"Central Difference: {central_difference(f, x0)}")
    print(f"Second Derivative: {second_derivative(f, x0)}")

    solve_heat_equation_1d(dx=0.05, dt=0.001, T=0.5, L=1.0, alpha=1.0, plot=True)

    solve_wave_equation_1d(dx=0.05, dt=0.001, T=1.0, L=1.0, c=1.0, plot=True)
    solve_wave_equation_1d()
    