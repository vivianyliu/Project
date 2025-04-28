from method_finite_element import (
    generate_mesh,
    assemble_stiffness_matrix,
    assemble_load_vector,
    generate_2d_mesh,
    assemble_2d_stiffness,
    assemble_2d_load_vector,
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

if __name__ == '__main__':
    unittest.main()
