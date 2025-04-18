from hermite_rbf_dq import solve_fractional_rbf_dq_2D
import matplotlib.pyplot as plt

X, Y, U = solve_fractional_rbf_dq_2D(N=20, T=0.2, dt=0.01)
plt.contourf(X, Y, U, levels=50, cmap='viridis')
plt.colorbar(label="u(x, y, T)")
plt.title("H-RBF-DQ Solution at Final Time (Fractional)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()