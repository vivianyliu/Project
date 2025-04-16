import numpy as np
import matplotlib.pyplot as plt

def solve_lax_wendroff_advection(u0, a=1.0, L=1.0, T=1.0, Nx=200, CFL=0.8, nonlinear=False):
    """
    Solves the linear or nonlinear advection equation using Lax-Wendroff method.

    Parameters:
        u0: function, initial condition u(x, 0)
        a: float, advection speed
        L: float, domain length
        T: float, final time
        Nx: int, number of spatial points
        CFL: float, Courant number for stability
        nonlinear: bool, whether to use nonlinear form (Burgers' type)
        
    Returns:
        x: array of grid points
        u: 2D array of solution u(x, t)
    """
    dx = L / Nx
    x = np.linspace(0, L, Nx, endpoint=False)
    dt = CFL * dx / abs(a if not nonlinear else 1.0)
    Nt = int(T / dt)

    u = np.zeros((Nt+1, Nx))
    u[0, :] = u0(x)

    for n in range(0, Nt):
        un = u[n, :].copy()

        if nonlinear:
            f = 0.5 * un**2
            df = un
        else:
            f = a * un
            df = a

        u[n+1, 1:-1] = un[1:-1] - 0.5 * dt/dx * (f[2:] - f[:-2]) + \
                       0.5 * (dt/dx)**2 * (df[2:] + df[:-2]) * (un[2:] - 2*un[1:-1] + un[:-2])

        # Periodic BC
        u[n+1, 0] = u[n+1, -2]
        u[n+1, -1] = u[n+1, 1]
    return x, u

def plot_lax_wendroff(x, u, times, dt, title="Lax-Wendroff Advection Solution"):
    plt.figure(figsize=(10, 5))
    for t in times:
        plt.plot(x, u[int(t/dt)], label=f"t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    u0 = lambda x: np.exp(-100 * (x - 0.5)**2)
    x, u = solve_lax_wendroff_advection(u0, a=1.0, L=1.0, T=1.0, Nx=400, CFL=0.8, nonlinear=False)
    dt = 0.8 * (1.0 / 400)
    plot_lax_wendroff(x, u, times=[0, 0.25, 0.5, 0.75, 1.0], dt=dt)

    # Nonlinear case (Burgers-like)
    x, u = solve_lax_wendroff_advection(u0, a=1.0, L=1.0, T=1.0, Nx=400, CFL=0.5, nonlinear=True)
    dt = 0.5 * (1.0 / 400)
    plot_lax_wendroff(x, u, times=[0, 0.25, 0.5, 0.75, 1.0], dt=dt,
                      title="Nonlinear Advection (Lax-Wendroff)")
