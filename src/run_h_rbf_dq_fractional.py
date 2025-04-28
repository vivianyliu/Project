from method_hermite_rbf_dq import solve_2D_hermite_rbf_neumann
import matplotlib.pyplot as plt

X, Y, U_hermite = solve_2D_hermite_rbf_neumann()
plt.figure(figsize=(6,5))
plt.contourf(X, Y, U_hermite, levels=50, cmap='plasma')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Hermite RBF-DQ | Heat solution at T=0.01 with Neumann BCs')
plt.show()