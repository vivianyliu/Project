from method_lax_wendroff import lax_wendroff_linear, compute_phase_error, lax_wendroff_burgers
import unittest
import numpy as np

class TestLaxWendroff(unittest.TestCase):

    def test_lax_wendroff_linear_constant_wave(self):
        N = 100
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        u0 = np.sin(x)
        a = 1.0
        dx = x[1] - x[0]
        dt = 0.5 * dx / a  # CFL condition
        t_max = 2 * np.pi / a

        u_hist = lax_wendroff_linear(u0, a, dx, dt, t_max)
        self.assertEqual(u_hist.shape, (int(t_max / dt) + 1, N))

    def test_compute_phase_error_zero_for_exact_shift(self):
        N = 100
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        u0 = np.sin(x)
        a = 1.0
        dx = x[1] - x[0]
        dt = 0.5 * dx / a
        t_max = 2*np.pi / a

        u_hist = lax_wendroff_linear(u0, a, dx, dt, t_max)
        u_exact = np.sin(x)  # after full period sin(x) returns to itself

        shift = compute_phase_error(u_hist, u_exact)
        self.assertTrue(abs(shift) <= 1)  # allow small offset due to numerical errors

    def test_lax_wendroff_burgers_conservation(self):
        N = 100
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        u0 = np.sin(x)
        dx = x[1] - x[0]
        dt = 0.2 * dx  # smaller dt for nonlinear case
        t_max = 0.5

        u_hist = lax_wendroff_burgers(u0, dx, dt, t_max)
        mass_initial = np.sum(u_hist[0])
        mass_final = np.sum(u_hist[-1])

        self.assertAlmostEqual(mass_initial, mass_final, places=1)

if __name__ == '__main__':
    unittest.main()
