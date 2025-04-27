from method_finite_element import (
    generate_mesh,
    assemble_stiffness_matrix,
    assemble_load_vector,
    solve_poisson_fem,
    generate_2d_mesh,
    assemble_2d_stiffness,
    assemble_2d_load_vector,
    solve_poisson_2d,
)
import unittest
import numpy as np

class TestFiniteElement(unittest.TestCase):

    def test_generate_mesh(self):
        L = 1.0
        nx = 4
        nodes, elements = generate_mesh(L, nx)
        expected_nodes = np.linspace(0, L, nx + 1)
        self.assertTrue(np.allclose(nodes, expected_nodes))
        self.assertEqual(elements.shape, (nx, 2))

    def test_assemble_stiffness_matrix(self):
        nodes = np.array([0.0, 0.5, 1.0])
        elements = np.array([[0,1],[1,2]])
        K = assemble_stiffness_matrix(nodes, elements)
        self.assertEqual(K.shape, (3,3))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T)) # check symmetry

    def test_assemble_load_vector(self):
        nodes = np.array([0.0, 0.5, 1.0])
        elements = np.array([[0,1],[1,2]])
        f = lambda x: 2.0
        F = assemble_load_vector(nodes, elements, f)
        self.assertEqual(len(F), 3)
        self.assertTrue(np.all(F >= 0)) # Load vector should be non-negative if f positive

    def test_solve_poisson_fem(self):
        nodes, u = solve_poisson_fem(L=1.0, nx=10, f=lambda x: 1.0)
        self.assertEqual(len(nodes), len(u))
        self.assertAlmostEqual(u[0], 0.0, places=6) # Boundary condition u(0) = 0
        self.assertAlmostEqual(u[-1], 0.0, places=6) # Boundary condition u(1) = 0

    def test_generate_2d_mesh(self):
        points, triangles = generate_2d_mesh(5, 5)
        self.assertTrue(points.shape[1] == 2)
        self.assertTrue(triangles.shape[1] == 3)

    def test_assemble_2d_stiffness(self):
        points, triangles = generate_2d_mesh(5, 5)
        K = assemble_2d_stiffness(points, triangles)
        self.assertEqual(K.shape[0], len(points))
        self.assertTrue(np.allclose(K.toarray(), K.toarray().T)) # should be symmetric

    def test_assemble_2d_load_vector(self):
        points, triangles = generate_2d_mesh(5, 5)
        f = lambda x, y: x + y
        F = assemble_2d_load_vector(points, triangles, f)
        self.assertEqual(len(F), len(points))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_solve_poisson_2d(self):
        points, triangles, u = solve_poisson_2d(nx=5, ny=5, f=lambda x, y: 1.0)
        self.assertEqual(len(points), len(u))
        # Check that Dirichlet boundary conditions hold approximately (should be 0)
        boundary_nodes = [i for i, (x,y) in enumerate(points) if x == 0 or x == 1 or y == 0 or y == 1]
        for i in boundary_nodes:
            self.assertAlmostEqual(u[i], 0.0, places=3)

if __name__ == '__main__':
    unittest.main()
