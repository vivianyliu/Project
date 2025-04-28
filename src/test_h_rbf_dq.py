from method_hermite_rbf_dq import (
    gaussian_rbf,
    construct_rbf_dq_matrix,
    construct_rbf_dq_derivatives,
    get_l1_weights,
    solve_rbf_dq_2D,
    solve_fractional_rbf_dq_2D
)
import unittest
import numpy as np

class TestHermiteRBFDQ(unittest.TestCase):

    def test_gaussian_rbf(self):
        r = np.array([0, 1, 2])
        values = gaussian_rbf(r, epsilon=1.0)
        expected = np.exp(-r**2)
        np.testing.assert_allclose(values, expected, atol=1e-8)

    def test_construct_rbf_dq_matrix(self):
        nodes = np.random.rand(5, 2)
        D = construct_rbf_dq_matrix(nodes, epsilon=1.0)
        self.assertEqual(D.shape, (5, 5))

    def test_construct_rbf_dq_derivatives(self):
        nodes = np.random.rand(5, 2)
        Dx, Dy, Lap = construct_rbf_dq_derivatives(nodes, epsilon=1.0)
        self.assertEqual(Dx.shape, (5, 5))
        self.assertEqual(Dy.shape, (5, 5))
        self.assertEqual(Lap.shape, (5, 5))

    def test_get_l1_weights(self):
        alpha = 0.5
        dt = 0.01
        N_steps = 10
        weights = get_l1_weights(alpha, dt, N_steps)
        self.assertEqual(len(weights), N_steps)

    def test_solve_rbf_dq_2D(self):
        X, Y, u = solve_rbf_dq_2D(N=10, T=0.1, dt=0.01, epsilon=1.0)
        self.assertEqual(u.shape, (10, 10))

    def test_solve_fractional_rbf_dq_2D(self):
        X, Y, u = solve_fractional_rbf_dq_2D(N=10, T=0.05, dt=0.01, D=0.01, vx=1.0, vy=1.0)
        self.assertEqual(u.shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
