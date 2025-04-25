from finite_element import solve_poisson_2d
import matplotlib.pyplot as plt

points, triangles, u = solve_poisson_2d()

plt.tripcolor(points[:, 0], points[:, 1], triangles, u, shading='gouraud')
plt.colorbar(label='u(x, y)')
plt.title("2D FEM Solution to -Δu = 1 on [0,1]²")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()