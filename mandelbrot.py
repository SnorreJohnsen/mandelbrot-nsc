"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics, os
import matplotlib.pyplot as plt
from numba import njit, prange
from multiprocessing import Pool

def row_sums(A: np.ndarray) -> float:
    """ computes row sums of square matrix"""
    for i in range(len(A)):     # len(A) gives number of first row
        s = np.sum(A[i, :])

    return s

def column_sums(A: np.ndarray) -> float:
    """ computes column sums of square matrix"""
    for j in range(len(A)):     # len(A) gives number of first row
        s = np.sum(A[:, j])

    return s
    
def mandelbrot_point_naive ( c ) :
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

def compute_mandelbrot_naive (x_min, x_max, y_min, y_max, resx, resy):
    """
    compute mandelbrot set over 2d region naive implementation

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
            all_n[i, j] = mandelbrot_point_naive(c)
    return all_n

def compute_mandelbrot_vectorized(x_min, x_max, y_min, y_max, resx, resy):
    """
    compute mandelbrot set over 2d region using numpy vectorized

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
    M : numpy.ndarray of shape (resx, resy)
        2d array containing number of iterations before magnitude grows too large for each point in the complex grid
    """
    # Parameter max iteration if not exceding 2
    max_iter = 100

    #create evenly spaced numbers
    x = np.linspace(x_min, x_max, resx)
    y = np.linspace(y_min, y_max, resy)

    # create meshgrid
    X, Y = np.meshgrid(x, y)

    C = X + 1j*Y

    # init Z (complex) and M (iteration counter) arrays shape like C
    Z = np.zeros_like(C, dtype=complex)
    M = np.zeros_like(C, dtype=int)

    # Compute mandelbrot point function vectorized
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2               # Boolean mask True: not diverged yet, False: points have escaped
        Z[mask] = Z[mask]**2 + C[mask]      # Updates only Z if mask True
        M[mask] += 1                        # Counts up max iteration matrix

    return M

@njit
def mandelbrot_point_numba(c):
    z = 0j 
    max_iter = 100

    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z**2 + c
    return n

@njit
def compute_mandelbrot_numba(x_min, x_max, y_min, y_max, resx, resy):
    """
    compute mandelbrot set over 2d region numba implementation

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
    all_n = np.zeros((resx, resy), dtype = np.int32)

    for i in range(resx):
        for j in range(resy):
            c = x[i] + 1j * y[j]
            all_n[i, j] = mandelbrot_point_numba(c)
    return all_n

# mandelbrot_parallel.py (Tasks 1-3 are one continuous script)
@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_real_sq = z_real*z_real
        z_imag_sq = z_imag*z_imag
        if z_real_sq + z_imag_sq > 4.0:
            return i
        z_imag_new = 2.0*z_real*z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        z_imag = z_imag_new
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)
    
def benchmark (func,
               *args, 
               n_runs =3) :
    """ Time func , return median of n_runs . From slides. """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func (* args )
        times.append(time.perf_counter () - t0 )
    median_t = statistics.median(times)
    print (f"Median : {median_t:.4f} s"
           f" ( min ={min(times):.4f} , max ={max(times):.4f})" )
    return median_t , result

# Example of usage
# t , M = benchmark ( my_mandelbrot , -2 , 1 , -1.5 , 1.5 , 1024 , 1024 , 100)


if __name__ == "__main__":

    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 100

    _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter) # warmup run
    t_serial, mb_serial = benchmark(mandelbrot_serial,
                                    N, x_min, x_max, y_min, y_max, max_iter,
                                    n_runs=3)
    
    print(f"Computation took {t_serial:.3f} seconds")


    #to crate image of mandelbrot
    plt.imshow(mb_serial, cmap = "hot")
    plt.title("Mandelbrot plot vectorized")
    plt.colorbar()
    plt.show()
  