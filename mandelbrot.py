"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def mandelbrot_point ( c ) :
    """
    computes one mandelbrot point

    Parameters
    ----------
    c : complex
        complex constant
    
    
    Returns
    -------
    n : int 
        iterations
    """

    z = 0 
    max_iter = 100

    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n
    return n


def compute_mandelbrot (x_min, x_max, y_min, y_max, resx, resy):
    """
    compute mandelbrot set over 2d region

    Parameters
    ----------
    x_min : float
        minimum real value of region
    x_max : float
        maximum real value of region 
    y_min : float
        minimum imaginary value of region
    y_max : float
        maximum imaginary value of region 
    resx : int
        number of points in x-axis
    resy : int
        number of points in y-axis
    
    Returns
    -------
    all_n : numpy.ndarray of shape (resx, resy)
        2d array containing number of iterations before magnitude grows too large for each point in the complex grid
    """

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
    start = time.time()
    all_n = compute_mandelbrot(-2, 1, -1.5, 1.5, 1024, 1024)
    elapsed = time.time() - start
    print(f"Computation took {elapsed:.3f} seconds")

    #to crate image of mandelbrot
    plt.imshow(all_n, cmap = "hot")
    plt.title("Mandelbrot plot")
    plt.colorbar()
    plt.savefig("naive_mandelbrot.png")
    plt.show()
  