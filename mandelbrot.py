"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time

def mandelbrot_point ( c ) :
    """
    Example function .
    Parameters
    ----------
    c : float
    Input value
    Returns
    -------
    float
    Output value
    """

    z = 0 
    max_iter = 100

    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n
    return n

def compute_mandelbrot (x_min, x_max, y_min, y_max, resx, resy):


    #create evenly spaced numbers
    x = np.linspace(x_min, x_max, resx)
    y = np.linspace(y_min, y_max, resy)

    #create arrays for iterations
    all_n = np.zeros((resx, resy), dtype = int)

    for i in range(resx):
        for j in range(resy):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point(c)
    return all_n
            



if __name__ == "__main__":

    all_n = compute_mandelbrot(-2, 1, -1.5, 1.5, 100, 100)