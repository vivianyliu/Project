import numpy as np
import matplotlib.pyplot as plt
import math
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
        D: derivative approximation matrix (mocked)
    """
    r = cdist(nodes, nodes)
    A = gaussian_rbf(r, epsilon)
    A_inv = np.linalg.pinv(A)
    D = np.dot(A_inv, np.gradient(A, axis=0))  # crude placeholder
    return D

def construct_rbf_dq_derivatives(nodes, epsilon=2.0):
    """
    Constructs RBF-DQ matrices for ∂/∂x, ∂/∂y, and Laplacian ∇².

    Parameters:
        nodes: array of shape (N, 2)
        epsilon: shape parameter

    Returns:
        Dx, Dy, Lap: derivative matrices
    """
    r = cdist(nodes, nodes)
    A = gaussian_rbf(r, epsilon)
    A_inv = np.linalg.pinv(A)

    dx = nodes[:, 0][:, None] - nodes[:, 0]
    dy = nodes[:, 1][:, None] - nodes[:, 1]

    dA_dx = -2 * epsilon**2 * dx * A
    dA_dy = -2 * epsilon**2 * dy * A
    d2A = 2 * epsilon**2 * (2 * epsilon**2 * (dx**2 + dy**2) - 1) * A

    Dx = A_inv @ dA_dx
    Dy = A_inv @ dA_dy
    Lap = A_inv @ d2A
    return Dx, Dy, Lap

def get_l1_weights(alpha, dt, N_steps):
    """Generates L1 weights for time-fractional Caputo derivative."""
    gamma = np.zeros(N_steps)
    gamma[0] = 1
    for k in range(1, N_steps):
        gamma[k] = (1 - (1 + alpha) / k) * gamma[k - 1]
    weights = dt**(-alpha) * gamma / math.gamma(2 - alpha)
    return weights


def solve_2D_rbf_dq(N=10, T=1.0, dt=0.01, epsilon=1.0):
    """
    Solves a 2D diffusion-type equation using H-RBF-DQ.

    Parameters:
        N: number of nodes along each axis
        T: final time
        dt: time step
        epsilon: RBF shape parameter
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

def solve_fractional_rbf_dq_2D(N=20, T=0.1, dt=0.01, D=0.01, vx=1.0, vy=1.0):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2)).ravel()

    Dx, Dy, Lap = construct_rbf_dq_derivatives(nodes, epsilon=2.0)

    Nt = int(T / dt)
    U_hist = [u.copy()]
    a_grid = 0.8 + 0.2 * np.sin(2 * np.pi * nodes[:, 0])  # variable order α(x)

    all_weights = [get_l1_weights(a, dt, Nt) for a in a_grid]

    for n in range(1, Nt):
        u_new = np.zeros_like(u)

        for i in range(len(u)):
            weights = all_weights[i][:n] # truncate weights to available history
            available_history = np.array([U_hist[n - k] for k in range(1, n + 1)])
            frac_term = np.dot(weights, available_history[:, i])

            advection = -vx * (Dx @ u)[i] - vy * (Dy @ u)[i]
            diffusion = D * (Lap @ u)[i]
            source = 0

            u_new[i] = frac_term + dt**a_grid[i] * (diffusion + advection + source)

        U_hist.append(u_new.copy())
        u = u_new.copy()

    return X, Y, u.reshape(N, N)
