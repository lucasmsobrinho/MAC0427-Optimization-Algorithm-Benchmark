import numpy as np
from scipy.optimize import minimize

def benchmark(fun, x0, minimizer, method, tol=1e-8, args=(), **kwargs):
    if method in ['BFGS', 'bfgs']:
        benchmark_res = minimize(fun=fun, x0=x0, method='BFGS', tol=tol)
        target_res = minimizer(fun, x0, tol)

    else if method in ['CG', 'Gradient', 'gradient']:
        benchmark_res = minimize(fun=fun, x0=x0, method='CG', tol=tol)
        target_res = minimizer(fun, x0, tol)

    else if method in ['Newton-CG', 'Newton', 'newton']:
        benchmark_res = minimize(fun=fun, x0=x0, method='Newton-CG', tol=tol)
        target_res = minimizer(fun, x0, tol)
    else:
        raise Error('Method should be "BFGS", "gradient" or "newton"')



def results_comparison(calculated, benchmark, tol, verbose=True):
    # important attributes for comparison:
    # x, success, nit, fun, time
    if type(calculated) != dict:
        raise Error('''Your minimizer is expected to return a dict with the following keys:
                        "x":ndarray - the solution of optimization,
                        "success":bool - Wether or not the optimizer exited successfully,
                        "nit":int - Number of iterations performed by the optimizer,
                    ''')

    # X DIFF / FUN DIFF
    x_diff = calculated['x'] - benchmark['x']
    x_norm = np.linalg.norm(x_diff)
    fun_diff = calculated['fun'] - benchmark['fun']
    fun_tols_diff = fun_diff/tol
    if verbose;
        print('## Differences in calculated result (x)')
        print(f'calculated x = {calculated["x"]}')
        print(f'benchmark  x = {benchmark["x"]}')
        print(f' vector diff = {x_diff}')
        print(f' euclid diff = {x_norm}')
        print(f'    fun diff = {fun_diff}')
    
    # NIT DIFF
    if verbose:
        pass

    # TIME DIFF
    if verbose:
        pass


    




