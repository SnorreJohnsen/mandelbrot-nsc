"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics
import matplotlib.pyplot as plt

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
    N = 10000
    A = np.random.rand(N, N)
    A_f = np.asfortranarray(A)

    time_row, _ = benchmark(row_sums, A_f)
    time_column, _ = benchmark(column_sums, A_f)

    print(f"{time_row=}")
    print(f"{time_column=}")

    # Result for nparray 
    # Median : 0.1195 s ( min =0.1182 , max =0.8294)
    # Median : 1.4539 s ( min =1.4427 , max =1.5742)
    # time_row=0.11948005000158446
    # time_column=1.4538839970009576

    # Result for fortran arrray
    # Median : 1.3178 s ( min =1.2459 , max =1.5608)
    # Median : 0.0989 s ( min =0.0973 , max =0.1113)
    # time_row=1.317769376000797
    # time_column=0.0988852999980736

    exit()

    # Benchmark naive
    #elapsed_time_nai, mandelbrot_nai_out = benchmark(compute_mandelbrot_naive, -2, 1, -1.5, 1.5, 1024, 1024, n_runs=1)
    #print(f"Computation took {elapsed_time_nai:.3f} seconds")

    # Benchmark vectorized
    elapsed_time_vec, mandelbrot_vec_out = benchmark(compute_mandelbrot_vectorized, -2, 1, -1.5, 1.5, 1024, 1024, n_runs=1)
    print(f"Computation took {elapsed_time_vec:.3f} seconds")


    #to crate image of mandelbrot
    plt.imshow(mandelbrot_vec_out, cmap = "hot")
    plt.title("Mandelbrot plot vectorized")
    plt.colorbar()
    plt.savefig("vectorized_mandelbrot.png")
    plt.show()
  