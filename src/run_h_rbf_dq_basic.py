from method_hermite_rbf_dq import solve_rbf_dq_2D
import matplotlib.pyplot as plt

X, Y, U = solve_rbf_dq_2D(N=20, T=0.1, dt=0.005, epsilon=2.0)
plt.contourf(X, Y, U, levels=50, cmap='plasma')
plt.title("2D H-RBF-DQ Solution Snapshot")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="u(x, y, T)")
plt.grid(False)
plt.show()