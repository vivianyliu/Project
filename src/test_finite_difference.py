from method_finite_difference import forward_difference, backward_difference, central_difference, second_derivative
import unittest
import numpy as np

class TestFiniteDifference(unittest.TestCase):
    
    def test_forward_difference(self):
        f = lambda x: np.sin(x)
        df = lambda x: np.cos(x)
        x0 = np.pi / 4
        approx = forward_difference(f, x0)
        exact = df(x0)
        self.assertAlmostEqual(approx, exact, places=5)
    
    def test_backward_difference(self):
        f = lambda x: np.sin(x)
        df = lambda x: np.cos(x)
        x0 = np.pi / 4
        approx = backward_difference(f, x0)
        exact = df(x0)
        self.assertAlmostEqual(approx, exact, places=5)

    def test_central_difference(self):
        f = lambda x: np.sin(x)
        df = lambda x: np.cos(x)
        x0 = np.pi / 4
        approx = central_difference(f, x0)
        exact = df(x0)
        self.assertAlmostEqual(approx, exact, places=6)

    def test_second_derivative(self):
        f = lambda x: np.sin(x)
        d2f = lambda x: -np.sin(x)
        x0 = np.pi / 4
        approx = second_derivative(f, x0)
        exact = d2f(x0)
        self.assertAlmostEqual(approx, exact, places=4)

if __name__ == '__main__':
    unittest.main()
