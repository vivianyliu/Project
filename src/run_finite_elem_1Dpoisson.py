from method_finite_element import generate_mesh, assemble_stiffness_matrix, assemble_load_vector, apply_boundary_conditions
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def solve_poisson_1d(L=1.0, nx=10, f=lambda x: 1.0):
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

nodes, u = solve_poisson_1d(L=1.0, nx=20, f=lambda x: 1.0)
plt.plot(nodes, u, "-o", label="FEM Solution")
plt.title("1D FEM Solution to Poisson Equation, f(x) = 1")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.show()