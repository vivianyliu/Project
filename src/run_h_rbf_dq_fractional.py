from method_hermite_rbf_dq import solve_2D_hermite_rbf_neumann
import matplotlib.pyplot as plt

X, Y, U_hermite = solve_2D_hermite_rbf_neumann()
plt.figure(figsize=(6,5))
plt.contourf(X, Y, U_hermite, levels=50, cmap='plasma')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fractional H-RBF-DQ Heat Solution at T=0.01, Neumann BC')
plt.show()