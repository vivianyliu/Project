import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def gaussian_rbf(r, epsilon=1.0):
    """
    Gaussian radial basis function.

    Parameters:
        r: distances between nodes
        epsilon: shape parameter

    Returns:
        RBF values
    """
    return np.exp(-(epsilon * r)**2)

def gaussian_rbf_derivatives(xi, xj, epsilon=1.0):
    """
    Computes the derivatives of Gaussian RBF with respect to x and y.

    Parameters:
        xi: evaluation points (N x 2)
        xj: center points (N x 2)
        epsilon: shape parameter

    Returns:
        A: RBF matrix
        dA_dx: derivative with respect to x
        dA_dy: derivative with respect to y
        d2A: Laplacian (second derivatives combined)
    """
    
    dx = xi[:, 0][:, None] - xj[:, 0]
    dy = xi[:, 1][:, None] - xj[:, 1]
    r = np.sqrt(dx**2 + dy**2)
    A = gaussian_rbf(r, epsilon)
    dA_dx = -2 * epsilon**2 * dx * A
    dA_dy = -2 * epsilon**2 * dy * A
    d2A = 2 * epsilon**2 * (2 * epsilon**2 * (dx**2 + dy**2) - 1) * A
    return A, dA_dx, dA_dy, d2A

def laplacian_gaussian_rbf(r, dx, dy, epsilon=1.0):
    """
    Compute Laplacian of Gaussian RBF.

    Parameters:
        r: distances between nodes
        dx: x-distances between nodes
        dy: y-distances between nodes
        epsilon: shape parameter

    Returns:
        Laplacian of Gaussian RBF
    """
    
    r2 = r**2
    A = gaussian_rbf(r, epsilon)
    term = 2 * epsilon**2 * (2 * epsilon**2 * r2 - 1)
    return term * A

def construct_rbf_dq_laplacian(nodes, epsilon=1.0):
    """
    Constructs Laplacian operator matrix using standard RBF-DQ.

    Parameters:
        nodes: array of shape (N, 2)
        epsilon: shape parameter

    Returns:
        Lap: Laplacian operator matrix
    """
    
    r = cdist(nodes, nodes)
    A = gaussian_rbf(r, epsilon)
    A_inv = np.linalg.pinv(A)

    dx = nodes[:, 0][:, None] - nodes[:, 0]
    dy = nodes[:, 1][:, None] - nodes[:, 1]

    d2A = 2 * epsilon**2 * (2 * epsilon**2 * (dx**2 + dy**2) - 1) * A
    Lap = A_inv @ d2A
    return Lap

def construct_hermite_rbf_laplacian(nodes, epsilon=1.0):
    """
    Constructs Laplacian operator matrix using Hermite-corrected RBF-DQ.

    Parameters:
        nodes: array of shape (N, 2)
        epsilon: shape parameter

    Returns:
        Lap: corrected Laplacian operator matrix
    """
    r = cdist(nodes, nodes)
    dx = nodes[:, 0][:, None] - nodes[:, 0]
    dy = nodes[:, 1][:, None] - nodes[:, 1]

    A = gaussian_rbf(r, epsilon)
    A_inv = np.linalg.pinv(A)

    Lap_A = laplacian_gaussian_rbf(r, dx, dy, epsilon)

    Lap = A_inv @ Lap_A
    return Lap

def solve_2D_hermite_rbf_neumann(N=50, T=0.01, dt=0.001, epsilon=1.0):
    """
    Solves the 2D heat equation using Hermite RBF-DQ Laplacian with Neumann boundary conditions.

    Parameters:
        N: number of points per axis
        T: final time
        dt: time step
        epsilon: shape parameter

    Returns:
        X: x-grid points
        Y: y-grid points
        u: solution at final time (N x N)
    """
    
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    # Initial condition: centered Gaussian
    u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2)).ravel()

    Lap = construct_hermite_rbf_laplacian(nodes, epsilon)

    Nt = int(T / dt)

    # Solve the heat equation with Neumann boundary conditions
    for _ in range(Nt):
        # Update the solution using forward Euler, with Neumann conditions
        u_new = u + dt * (Lap @ u)

        # Enforce Neumann boundary conditions (derivative = 0 at boundaries)
        u_new[0:N] = u_new[N:2*N]  # y = 0 boundary (no flux in y-direction)
        u_new[(N-1)*N:N*N] = u_new[(N-2)*N:(N-1)*N]  # y = 1 boundary (no flux in y-direction)
        u_new[::N] = u_new[1::N]  # x = 0 boundary (no flux in x-direction)
        u_new[N-1::N] = u_new[N-2::N]  # x = 1 boundary (no flux in x-direction)

        # Update u with the new solution
        u = u_new

    return X, Y, u.reshape(N, N)


def solve_heat_rbf(N=50, T=0.01, dt=0.001, epsilon=0.5):
    """
    Solves the 2D heat equation using standard RBF-DQ Laplacian.

    Parameters:
        N: number of points per axis
        T: final time
        dt: time step
        epsilon: shape parameter

    Returns:
        X: x-grid points
        Y: y-grid points
        u: solution at final time (N x N)
    """
    
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T

    # Initial condition: centered Gaussian
    u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2)).ravel()

    Lap = construct_rbf_dq_laplacian(nodes, epsilon)

    Nt = int(T / dt)

    for _ in range(Nt):
        u += dt * (Lap @ u)  # Forward Euler

    return X, Y, u.reshape(N, N)