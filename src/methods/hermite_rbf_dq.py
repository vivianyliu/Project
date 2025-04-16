import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def gaussian_rbf(r, epsilon=1.0):
    return np.exp(-(epsilon * r)**2)

def construct_rbf_dq_matrix(nodes, epsilon=1.0):
    """
    Constructs the RBF-DQ derivative matrix using Gaussian RBFs.

    Parameters:
        nodes: array of shape (N, 2)
        epsilon: shape parameter

    Returns:
        D: derivative approximation matrix (mocked for now)
    """
    r = cdist(nodes, nodes)
    A = gaussian_rbf(r, epsilon)
    A_inv = np.linalg.pinv(A)
    D = np.dot(A_inv, np.gradient(A, axis=0))  # crude placeholder
    return D

def solve_2D_rbf_dq(N=10, T=1.0, dt=0.01, epsilon=1.0):
    """
    Solves a 2D diffusion-type equation using H-RBF-DQ.

    Parameters:
        N: number of nodes along each axis
        T: final time
        dt: time step
        epsilon: RBF shape parameter

    Returns:
        X, Y, u: spatial mesh and solution
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2)).ravel()

    D = construct_rbf_dq_matrix(nodes, epsilon)

    Nt = int(T / dt)
    for _ in range(Nt):
        u += dt * D @ u # simplified time steps u_t = D @ u

    return X, Y, u.reshape(N, N)


if __name__ == "__main__":
    X, Y, U = solve_2D_rbf_dq(N=20, T=0.1, dt=0.005, epsilon=2.0)
    plt.contourf(X, Y, U, levels=50, cmap='plasma')
    plt.title("H-RBF-DQ Solution Snapshot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="u(x, y, T)")
    plt.grid(False)
    plt.show()
