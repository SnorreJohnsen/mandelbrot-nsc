"""
Mandelbrot Set Generator
Author : [ Snorre Johnsen ]
Course : Numerical Scientific Computing 2026
"""



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
    return max_iter


if __name__ == "__main__":
    c = 0
    n = mandelbrot_point(c)

    print(f"{c=}", "and iterations =", n)