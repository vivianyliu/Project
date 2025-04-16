import numpy as np
import matplotlib.pyplot as plt

def lax_wendroff_linear(u0, a, dx, dt, t_max):
    """
    Solves the 1D linear advection equation using the second-order Lax-Wendroff scheme:
        u_t + a u_x = 0

    Parameters:
        u0     : array-like, initial condition sampled on the grid
        a      : float, wave speed (positive or negative)
        dx     : float, spatial step size
        dt     : float, time step size
        t_max  : float, final simulation time

    Returns:
        u_hist : 2D array of shape (Nt+1, N), numerical solution at each time step
    """
    u = u0.copy()
    u_hist = [u.copy()]
    r = a * dt / dx
    steps = int(t_max / dt)

    for _ in range(steps):
        u_new = np.zeros_like(u)
        u_new[1:-1] = (u[1:-1]
                       - 0.5 * r * (u[2:] - u[:-2])
                       + 0.5 * r**2 * (u[2:] - 2*u[1:-1] + u[:-2]))
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        u = u_new.copy()
        u_hist.append(u.copy())
    return np.array(u_hist)

def compute_phase_error(u_hist, u_exact):
    """
    Computes the phase shift between the numerical and exact solution by comparing
    the location of the peak (maximum value) in both.

    Parameters:
        u_hist  : 2D array, numerical solution at all time steps
        u_exact : array-like, exact solution at final time

    Returns:
        shift   : int, index shift between numerical and exact peaks
    """
    u_final = u_hist[-1]
    shift = np.argmax(u_final) - np.argmax(u_exact)
    return shift

def lax_wendroff_burgers(u0, dx, dt, t_max):
    """
    Solves the nonlinear inviscid Burgers' equation using the Lax-Wendroff method:
        u_t + (u^2 / 2)_x = 0

    Parameters:
        u0     : array-like, initial condition sampled on the grid
        dx     : float, spatial step size
        dt     : float, time step size
        t_max  : float, final simulation time

    Returns:
        u_hist : 2D array of shape (Nt+1, N), numerical solution at each time step
    """
    u = u0.copy()
    u_hist = [u.copy()]
    steps = int(t_max / dt)

    for _ in range(steps):
        u_new = np.zeros_like(u)
        f = 0.5 * u**2 # Compute flux f = u^2 / 2 and its derivative f' as fp
        fp = u

        # L-W update
        u_new[1:-1] = u[1:-1] - (dt / (2 * dx)) * (f[2:] - f[:-2]) \
                      + (dt**2 / (2 * dx**2)) * (fp[2:] * (f[2:] - f[1:-1]) - fp[:-2] * (f[1:-1] - f[:-2]))

        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]

        u = u_new.copy()
        u_hist.append(u.copy())
    return np.array(u_hist)


def plot_solution_comparison(x, u_hist, u_exact, t, label="Numerical"):
    """
    Plots the numerical and exact solutions at a given time.

    Parameters:
        x       : array-like, spatial grid points
        u_hist  : 2D array, numerical solution at all time steps
        u_exact : array-like, exact solution at final time
        t       : float, time of snapshot
        label   : str, label for the numerical method used
    """
    plt.plot(x, u_exact, '--', label='Exact')
    plt.plot(x, u_hist[-1], label=label)
    plt.title(f"Solution at t={t}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
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

    # Burgers' equation
    u0_burgers = np.sin(2 * np.pi * x)
    dt_burgers = 0.3 * dx / np.max(np.abs(u0_burgers))  # conservative CFL
    t_max_burgers = 0.5
    u_hist_burgers = lax_wendroff_burgers(u0_burgers, dx, dt_burgers, t_max_burgers)
    plt.plot(x, u0_burgers, '--', label="Initial")
    plt.plot(x, u_hist_burgers[-1], label="Lax-Wendroff (Burgers)")
    plt.title("Nonlinear Advection: Inviscid Burgers' Equation")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()
