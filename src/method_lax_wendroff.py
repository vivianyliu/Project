import numpy as np

def compute_phase_error(u_hist, u_exact):
    """
    Computes the phase shift between the numerical and exact solution by comparing
    the location of the peak (maximum value) in both.

    Parameters:
        u_hist: 2D array, numerical solution at all time steps
        u_exact: array-like, exact solution at final time

    Returns:
        shift: int, index shift between numerical and exact peaks
    """
    u_final = u_hist[-1]
    shift = np.argmax(u_final) - np.argmax(u_exact)
    return shift


def solve_laxwendroff_linear(u0, a, dx, dt, t_max):
    """
    Solves the 1D linear advection equation using the second-order Lax-Wendroff scheme:
        u_t + a u_x = 0

    Parameters:
        u0: array-like, initial condition sampled on the grid
        a: float, wave speed (positive or negative)
        dx: float, spatial step size
        dt: float, time step size
        t_max: float, final simulation time

    Returns:
        u_hist: 2D array of shape (Nt+1, N), numerical solution at each time step
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

def solve_laxwendroff_burgers(u0, dx, dt, t_max):
    """
    Solves the nonlinear inviscid Burgers' equation using the Lax-Wendroff method:
        u_t + (u^2 / 2)_x = 0

    Parameters:
        u0: array-like, initial condition sampled on the grid
        dx: float, spatial step size
        dt: float, time step size
        t_max: float, final simulation time

    Returns:
        u_hist: 2D array of shape (Nt+1, N), numerical solution at each time step
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
