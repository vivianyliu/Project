import numpy as np

def forward_difference(f, x, h=1e-6):
    """To find approximation of f'(x) by forward difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """

    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-6):
    """To find approximation of f'(x) by backward difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """

    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-6):
    """To compute f'(x) using the central difference method.
    More accurate than forward or backward difference.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-6)
    """
    
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    """Compute f''(x) using the second-order central difference method.
 
    Parameters:
    f (function): Function to differentiate
    x (float): Point at which to compute the derivative
    h (float): Step size (default: 1e-5)
    """
    
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def div_difference(x, y):
    """Compute divided differences where arrays have unequal step sizes.

    Parameters:
    x (array): First array
    y (array): Second array
    """

    n = x.size
    y1 = y.copy()
    dd = np.zeros(n)

    for i in range(n):
        dd[i] = y1[i]
    for j in range(1, n):
        for i in range(n - j):
            y1[i] = (y1[i + 1] - y1[i]) / (x[i + j] - x[i])
        dd[j] = y1[0]
    
    return (dd)

def interp_poly(xx, yy):
    """Returns the interpolation function when using divided differences.
 
    Parameters:
    x (array): First array
    y (array): Second array
    """

    dd = div_difference(xx, yy)
    def poly(x):
        n = xx.size
        p = dd[n-1]
        for i in range(n-2, -1, -1):
            p = p * (x-xx[i]) + dd[i]
        return (p)
    return (poly)
