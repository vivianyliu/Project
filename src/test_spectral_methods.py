from method_spectral_methods import solve_poisson_spectral_1D, solve_poisson_spectral_2D, compute_error_up_to_constant
import unittest
import numpy as np

class TestSpectralMethods(unittest.TestCase):

    def test_solve_poisson_spectral_1D(self):
        f = lambda x: np.sin(x)
        exact_solution = lambda x: np.sin(x)
        
        x, u_approx = solve_poisson_spectral_1D(f, N=256)
        u_exact = exact_solution(x)

        error = compute_error_up_to_constant(u_approx, u_exact)
        self.assertLess(error, 1e-4)

    def test_compute_error_up_to_constant(self):
        u_approx = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([2.0, 3.0, 4.0])
        error = compute_error_up_to_constant(u_approx, u_exact)
        self.assertAlmostEqual(error, 0.0, places=6)

    def test_solve_poisson_spectral_2D(self):
        f = lambda x, y: np.sin(x) * np.sin(y)
        X, Y, u_approx = solve_poisson_spectral_2D(f, N=64)
        # exact solution is sin(x)sin(y), up to a constant factor
        u_exact = np.sin(X) * np.sin(Y)
        error = compute_error_up_to_constant(u_approx, u_exact)
        self.assertLess(error, 1e-3)

if __name__ == '__main__':
    unittest.main()
