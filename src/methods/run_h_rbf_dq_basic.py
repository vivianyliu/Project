from hermite_rbf_dq import solve_2D_rbf_dq
import matplotlib.pyplot as plt

X, Y, U = solve_2D_rbf_dq(N=20, T=0.1, dt=0.005, epsilon=2.0)
plt.contourf(X, Y, U, levels=50, cmap='plasma')
plt.title("H-RBF-DQ Solution Snapshot")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="u(x, y, T)")
plt.grid(False)
plt.show()