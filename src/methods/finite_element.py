import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def generate_mesh(L, nx):
    """Generates a 1D mesh of equally spaced points.
    Returns node positions and element connectivity as np.ndarray 's.

    Parameters:
        L (float): Length of the domain.
        nx (int): Number of elements.
    """

    nodes = np.linspace(0, L, nx + 1)
    elements = np.array([[i, i + 1] for i in range(nx)])
    return nodes, elements

def assemble_stiffness_matrix(nodes, elements):
    """Assembles the global stiffness matrix using linear basis functions.
    Returns global stiffness matrix as scipy.sparse.csr_matrix.

    Parameters:
        nodes (np.ndarray): Node positions.
        elements (np.ndarray): Element connectivity.
    """

    n_nodes = len(nodes)
    K = sp.lil_matrix((n_nodes, n_nodes))

    for element in elements:
        i, j = element
        h = nodes[j] - nodes[i]

        # Local stiffness matrix for lin elements
        k_local = (1 / h) * np.array([[1, -1], [-1, 1]])
        # Assemble into global matrix K
        K[i:i+2, i:i+2] += k_local
    return K.tocsc()

def assemble_load_vector(nodes, elements, f):
    """Assembles global load vector.
    Returns the load vector as np.ndarray.

    Parameters:
        nodes (np.ndarray): Node positions.
        elements (np.ndarray): Element connectivity.
        f (function): Source function.
    """

    n_nodes = len(nodes)
    F = np.zeros(n_nodes)

    for elem in elements:
        i, j = elem
        h = nodes[j] - nodes[i]
        x_mid = (nodes[i] + nodes[j]) / 2 # Midpoint rule
        f_mid = f(x_mid)
        f_local = (h / 2) * np.array([f_mid, f_mid])
        F[i:i+2] += f_local
    return F

def apply_boundary_conditions(K, F, nodes, u0=0, uL=0):
    """Applies Dirichlet boundary conditions by modifying the system.
    Returns the modified matrix and vector as scipy.sparse.csr_matrix and np.ndarray.

    Parameters:
        K (scipy.sparse.csr_matrix): Stiffness matrix.
        F (np.ndarray): Load vector.
        nodes (np.ndarray): Node positions.
        u0 (float): Boundary condition at x=0.
        uL (float): Boundary condition at x=L.
    """

    # Apply u(0) = u0
    K[0, :] = 0
    K[0, 0] = 1
    F[0] = u0
    # Apply u(L) = uL
    K[-1, :] = 0
    K[-1, -1] = 1
    F[-1] = uL
    return K, F


def solve_poisson_fem(L=1.0, nx=10, f=lambda x: 1.0):
    """Solves the 1D Poisson equation -u''(x) = f(x) using FEM.
    Returns node positions and solution vector as np.ndarray 's.

    Parameters:
        L (float): Length of the domain.
        nx (int): Number of elements.
        f (function): Source function.
    """

    nodes, elements = generate_mesh(L, nx)
    K = assemble_stiffness_matrix(nodes, elements)
    F = assemble_load_vector(nodes, elements, f)

    # Apply Dirichlet boundary conditions
    K, F = apply_boundary_conditions(K, F, nodes, u0=0, uL=0)

    u = spla.spsolve(K, F)
    return nodes, u
