"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""
import time, statistics, os, psutil
import numpy as np
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

# --- MP2 M2: add below M1 in mandelbrot_parallel.py ---
def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
                        max_iter=100, n_workers=2):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)



    
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


    # --- MP2 M3: benchmark (in __main__ block) ---
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5

    # Serial baseline (Numba already warm after M1 warm-up)
    _ = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter) # warmup run
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    for n_workers in range(1, psutil.cpu_count(logical=False) + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(f"{n_workers:2d} workers: {t_par:.3f}s, "
            f"speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")



    exit()
    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 100
    n_workers = 2

    mb_par = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers) # warmup run
    
    #to crate image of mandelbrot
    plt.imshow(mb_par, cmap = "hot")
    plt.title("Mandelbrot plot vectorized")
    plt.colorbar()
    plt.show()
  