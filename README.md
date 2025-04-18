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

Finite Difference: classical grid-based method for solving PDEs using finite difference approximations of derivatives, used for time-dependent and steady-state problems with structured grids.
*Supports explicit and implicit schemes
*Straightforward to implement AND extend
*Ideal for problems with simple geometry and/or boundary conditions

Finite Element: variational method for approximating solutions to PDEs, usually on irregular domains / with complex boundary conditions.
*Local basis functions over elements (e.g., linear, quadratic)
*Supports adaptive meshing and unstructured grids
*Great for elliptic and structural mechanics problems

Spectral Methods: global approximation technique using trigonometric (FFT) or polynomial basis functions for high-accuracy.
*Efficient for periodic problems on rectangular domains
*Solves Poisson’s equation in 1D and 2D via spectral decomposition (see examples)
*Exhibits exponential convergence for smooth solutions

Lax-Wendroff Scheme: second-order accurate finite difference method for solving hyperbolic PDEs, especially advection equations.
*Handles both linear and nonlinear advection
*Captures wave propagation with reduced numerical dispersion
*Includes utilities for phase error and dispersion analysis

Hermite Radial Basis Function - Differential Quadrature (H-RBF-DQ): meshless method using radial basis functions and differential quadrature for high-order accuracy on scattered nodes.
*Solves 2D advection-diffusion and fractional time PDEs
*Supports variable-order Caputo fractional derivatives
*Great for handling irregular domains and complex dynamics without meshing

## Ongoing Work

- Add exact solutions for benchmarking fractional solvers
- Automate convergence testing
- Add PDE definitions and solvers in src/pdes/