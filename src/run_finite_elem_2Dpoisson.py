from method_finite_element import generate_2d_mesh, assemble_2d_stiffness, assemble_2d_load_vector, apply_dirichlet_bc_2d
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def solve_poisson_2d(nx=20, ny=20, f=lambda x, y: 1.0):
    """Solves the 2D Poisson equation -u'(x,y) = f(x,y) on [0,1]² using FEM.
    Returns node coordinates, triangle connectivity, and solution vector.

    Parameters:
        nx (int): Number of points in x-direction.
        ny (int): Number of points in y-direction.
        f (function): Source function f(x, y).
    """

    points, triangles = generate_2d_mesh(nx, ny)
    K = assemble_2d_stiffness(points, triangles)
    F = assemble_2d_load_vector(points, triangles, f)
    K, F = apply_dirichlet_bc_2d(K, F, points)
    u = spla.spsolve(K, F)
    return points, triangles, u

points, triangles, u = solve_poisson_2d()
plt.tripcolor(points[:, 0], points[:, 1], triangles, u, shading='gouraud')
plt.colorbar(label='u(x, y)')
plt.title("2D FEM Solution to Poisson, -Δu = 1 on [0,1]²")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()