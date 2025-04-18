# Project

21-366 Final Project Repo for vyliu and ajperez: 
Numerical Methods for Partial Differential Equations

A Python-based project exploring a range of numerical methods for solving linear and nonlinear partial differential equations (PDEs). This includes classical methods like Lax-Wendroff for hyperbolic problems, spectral methods for Poisson-type equations, and modern meshless approaches like Hermite Radial Basis Function-based Differential Quadrature (H-RBF-DQ), including fractional time derivatives.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Methods Implemented

- Finite Difference
Lax-Wendroff for 1D linear and nonlinear advection
Includes plotting utilities for solution snapshots and dispersion-induced phase shift error

- Spectral Methods
1D and 2D spectral Poisson solvers using FFTs
Periodic boundary conditions assumed

- Meshless Methods
Hermite RBF-DQ method with Gaussian basis
Handles diffusion and advection-diffusion equations, variable-order time-fractional Caputo derivatives, Neumann boundary conditions (planned extension)
Includes get_l1_weights() for computing fractional time stepping coefficients

- Finite Difference
Classical grid-based method for solving PDEs using finite difference approximations of derivatives, used for time-dependent and steady-state problems with structured grids.
    Supports explicit and implicit schemes
    Straightforward to implement AND extend
    Ideal for problems with simple geometry and/or boundary conditions

- Finite Element
Variational method for approximating solutions to PDEs, usually on irregular domains / with complex boundary conditions.
    Local basis functions over elements (e.g., linear, quadratic)
    Supports adaptive meshing and unstructured grids
    Great for elliptic and structural mechanics problems

- Spectral Methods
Global approximation technique using trigonometric (FFT) or polynomial basis functions for high-accuracy.
    Efficient for periodic problems on rectangular domains
    Solves Poisson’s equation in 1D and 2D via spectral decomposition (see examples)
    Exhibits exponential convergence for smooth solutions

- Lax-Wendroff Scheme
Second-order accurate finite difference method for solving hyperbolic PDEs, especially advection equations.
    Handles both linear and nonlinear advection
    Captures wave propagation with reduced numerical dispersion
    Includes utilities for phase error and dispersion analysis

- Hermite Radial Basis Function - Differential Quadrature (H-RBF-DQ)
Meshless method using radial basis functions and differential quadrature for high-order accuracy on scattered nodes.
    Solves 2D advection-diffusion and fractional time PDEs
    Supports variable-order Caputo fractional derivatives
    Great for handling irregular domains and complex dynamics without meshing

## Ongoing Work

- Add exact solutions for benchmarking fractional solvers
- Automate convergence testing
- Add PDE definitions and solvers in src/pdes/