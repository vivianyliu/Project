# import unittest
# import numpy as np
# from src.methods.finite_difference import forward_difference, backward_difference, central_difference, second_derivative

import os
import sys
import numpy as np
# from src.methods.finite_difference import forward_difference

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .methods.finite_difference import forward_difference

def test_forward_difference_on_sin():
    f = lambda x: np.sin(x)
    df_approx = forward_difference(f, np.pi/4)
    df_exact = np.cos(np.pi/4)
    assert np.isclose(df_approx, df_exact, atol=1e-5)

# class TestFiniteDifference(unittest.TestCase):
    
#     def test_forward_difference(self):
#         f = lambda x: np.sin(x)
#         df = lambda x: np.cos(x)
#         x0 = np.pi / 4
#         approx = forward_difference(f, x0)
#         exact = df(x0)
#         self.assertAlmostEqual(approx, exact, places=5)
    
#     def test_backward_difference(self):
#         f = lambda x: np.sin(x)
#         df = lambda x: np.cos(x)
#         x0 = np.pi / 4
#         approx = backward_difference(f, x0)
#         exact = df(x0)
#         self.assertAlmostEqual(approx, exact, places=5)

#     def test_central_difference(self):
#         f = lambda x: np.sin(x)
#         df = lambda x: np.cos(x)
#         x0 = np.pi / 4
#         approx = central_difference(f, x0)
#         exact = df(x0)
#         self.assertAlmostEqual(approx, exact, places=6)

#     def test_second_derivative(self):
#         f = lambda x: np.sin(x)
#         d2f = lambda x: -np.sin(x)
#         x0 = np.pi / 4
#         approx = second_derivative(f, x0)
#         exact = d2f(x0)
#         self.assertAlmostEqual(approx, exact, places=4)

# if __name__ == '__main__':
#     unittest.main()
