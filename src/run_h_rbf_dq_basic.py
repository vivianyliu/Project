from method_hermite_rbf_dq import solve_heat_rbf
import matplotlib.pyplot as plt

X, Y, u_normal = solve_heat_rbf()

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, u_normal, levels=50, cmap="plasma")
plt.colorbar()
plt.title("Basic RBF-DQ Heat Solution at T=0.01")
plt.xlabel("x")
plt.ylabel("y")