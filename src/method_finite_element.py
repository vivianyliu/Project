import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay

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


def generate_2d_mesh(nx, ny):
    """Generates a structured 2D mesh and triangulates it.
    Returns node coordinates and triangle connectivity.

    Parameters:
        nx (int): Number of points in x-direction.
        ny (int): Number of points in y-direction.
    """

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    tri = Delaunay(points)
    return points, tri.simplices

def local_stiffness_matrix(p1, p2, p3):
    """Computes the local stiffness matrix for a linear triangle.
    Returns the local matrix and area of the triangle.

    Parameters:
        p1, p2, p3 (np.ndarray): Coordinates of triangle vertices.
    """

    B = np.array([[p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]],
                  [p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]]])
    area = 0.5 * np.abs(np.linalg.det(np.array([[p2[0]-p1[0], p3[0]-p1[0]],
                                                [p2[1]-p1[1], p3[1]-p1[1]]])))
    return (B.T @ B) * (1 / (4 * area)), area

def assemble_2d_stiffness(points, triangles):
    """Assembles the global stiffness matrix for 2D Poisson problem.
    Returns the global stiffness matrix as scipy.sparse.csr_matrix.

    Parameters:
        points (np.ndarray): Node coordinates.
        triangles (np.ndarray): Triangle connectivity.
    """
    
    n_points = len(points)
    K = sp.lil_matrix((n_points, n_points))
    for tri in triangles:
        nodes = points[tri]
        k_local, _ = local_stiffness_matrix(*nodes)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += k_local[i, j]
    return K.tocsc()

def assemble_2d_load_vector(points, triangles, f):
    """Assembles the global load vector using midpoint quadrature.
    Returns the load vector as np.ndarray.

    Parameters:
        points (np.ndarray): Node coordinates.
        triangles (np.ndarray): Triangle connectivity.
        f (function): Source function f(x, y).
    """

    F = np.zeros(len(points))
    for tri in triangles:
        nodes = points[tri]
        _, area = local_stiffness_matrix(*nodes)
        centroid = np.mean(nodes, axis=0)
        f_val = f(centroid[0], centroid[1])
        F[tri] += f_val * area / 3 # Equal share to each node
    return F

def apply_dirichlet_bc_2d(K, F, points, boundary_func=lambda x, y: 0.0):
    """Applies Dirichlet boundary conditions on the square boundary.
    Modifies system matrix and RHS vector.

    Parameters:
        K (scipy.sparse.csr_matrix): Stiffness matrix.
        F (np.ndarray): Load vector.
        points (np.ndarray): Node coordinates.
        boundary_func (function): u(x, y) on boundary.
    """

    for i, (x, y) in enumerate(points):
        if x == 0 or x == 1 or y == 0 or y == 1:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = boundary_func(x, y)
    return K, F
