from method_finite_element import solve_poisson_fem
import matplotlib.pyplot as plt

# Solve Poisson equation with f(x) = 1
nodes, u = solve_poisson_fem(L=1.0, nx=20, f=lambda x: 1.0)
plt.plot(nodes, u, "-o", label="FEM Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Poisson Equation Solution")
plt.legend()
plt.grid()
plt.show()