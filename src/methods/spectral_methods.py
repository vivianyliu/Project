import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_spectral_1D(f, N=1000, L=2*np.pi):
    """
    Solves u''(x) = -f(x) on [0, L] with periodic BCs using spectral method.
    
    Parameters:
        f: function, the right-hand side f(x)
        N: int, number of grid points
        L: float, domain length
        
    Returns:
        x: array of grid points
        u: array of approximate solution at grid points
    """
    # Grid points
    x = np.linspace(0, L, N, endpoint=True)
    dx = L / N

    # Evaluate f(x) on the grid
    f_vals = f(x)

    # FFT of f
    f_hat = np.fft.fft(f_vals)

    # Wavenumbers
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_squared = -(k**2)
    k_squared[0] = 1  # avoid division by zero

    # Solve in spectral space
    u_hat = f_hat / k_squared
    u_hat[0] = 0  # set mean to zero

    # Inverse FFT to get u(x)
    u = np.fft.ifft(u_hat).real
    mean_shift = np.mean(-f(x)) - np.mean(u)
    u_aligned = u + mean_shift

    return x, u_aligned

def compute_error_up_to_constant(u_approx, u_exact):
    """
    Solves the 2 norm error of the approximated and exact solution
    
    Parameters:
        u_approx: the spectral solution
        u_exact: -f(x)
        
    Returns:
        l2_error: the 2 norm error of the spectral solution and -f(x)
    """
    # Calulates the difference in means
    shift = np.mean(u_exact) - np.mean(u_approx)

    # Updates aproximated solutin with +C correction
    u_aligned = u_approx + shift

    # Computes error
    abs_error = np.abs(u_aligned - u_exact)
    l2_error = np.sqrt(np.mean(abs_error**2))

    return l2_error

def solve_poisson_spectral_2D(f, N=1000, L=2*np.pi):
    """
    Solves u''(x) = -f(x, y) on [0, L]x[0, L] with periodic BCs using spectral method.
    
    Parameters:
        f: function, the right-hand side f(x, y)
        N: int, number of grid points in each dimension (NxN grid)
        L: float, domain length in x and y directions

    Returns:
        X, Y: 2D arrays of grid points in x and y
        u: 2D array of approximate solution u(x, y) at grid points   
    """
    # Grid points
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Evaluate f(x,y) on the grid
    f_vals = f(X, Y)

    # FFT2 of f
    f_hat = np.fft.fft2(f_vals)

    # Wavenumbers
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    k_squared = -(KX**2 + KY**2)
    k_squared[0, 0] = 1  # avoid division by zero

    # Solve in spectral space
    u_hat = f_hat / k_squared
    u_hat[0, 0] = 0  # remove mean

    # Inverse FFT2 to get u(x,y)
    u = np.fft.ifft2(u_hat).real
    
    return X, Y, u




if __name__ == "__main__":
    f = lambda x: np.exp(np.sin(x))
    x, u = solve_poisson_spectral_1D(f)
    err = compute_error_up_to_constant(u, -f(x))
    plt.plot(x, u, label='Spectral Solution')
    plt.plot(x, -f(x), '--', label='-f(x) =-e^(sin(x))', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title("Solution of $u''(x) = -e^{sin(x)}$ via Spectral Method")
    plt.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.3, -0.5, f'error ={err}', fontsize=8, bbox = props)
    plt.grid(True)
    plt.show()
    f = lambda x: np.sin(x)
    x, u = solve_poisson_spectral_1D(f)
    err = compute_error_up_to_constant(u, -f(x))
    plt.plot(x, u, label='Spectral Solution')
    plt.plot(x, -f(x), '--', label='-f(x) =-sin(x)', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title("Solution of $u''(x) = -sin(x)$ via Spectral Method")
    plt.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(3.5, -0.5, f'error ={err}', fontsize=8, bbox = props)
    plt.grid(True)
    plt.show()
    