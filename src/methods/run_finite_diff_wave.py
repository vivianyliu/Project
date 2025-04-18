from finite_difference import forward_difference, central_difference, second_derivative, solve_wave_equation_1d
import numpy as np

f = lambda x: np.sin(x)
x0 = np.pi / 4
print(f"Forward Difference: {forward_difference(f, x0)}")
print(f"Central Difference: {central_difference(f, x0)}")
print(f"Second Derivative: {second_derivative(f, x0)}")

solve_wave_equation_1d(dx=0.05, dt=0.001, T=1.0, L=1.0, c=1.0, plot=True)
solve_wave_equation_1d()