import unittest
import matplotlib.pyplot as plt
import numpy as np

import examples as examples
import src.unconstrained_min as unconstrained
from src.utils import *

def plot(X, Y, f, path_taken, title):
    params = list(zip(np.ravel(X), np.ravel(Y)))
    vals = []
    for p in params:
        vals.append(f(np.array(p))[0])
    Z = np.array(vals).reshape(X.shape[0], X.shape[1])
    fig, ax = plt.subplots(figsize=(30, 30))
    CS = ax.contour(X, Y, Z,5)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)
    for v in path_taken:
        plt.plot(v['location'][0], v['location'][1], 'x', c='red',ms=7)
    plt.show()
    
class TestUnconstrained(unittest.TestCase):

    def test_quadratic(self):
        methods = ['nt','bfgs','gd']
        for method in methods:
            print('test_quad_min with ' + method)
            x0 = np.array([1, 1])
            step_size = 0.02
            max_itr = 200
            step_tolerance = 10**-8
            obj_tolerance = 10**-10
            res = unconstrained.line_search(
                examples.quadratic_func_3,
                x0,
                step_size,
                obj_tolerance,
                step_tolerance,
                max_itr,
                method)
            X, Y = np.meshgrid(
                np.linspace(-5, 5),
                np.linspace(-5, 5),
                indexing='ij',
            )
            plot(
                X,
                Y,
                examples.quadratic_func_3,
                res['path_taken'],
                'quad min ' + method)
            plot_iterations_objective(res['path_taken'])
    def test_rosenbrock_min(self):
        methods = ['nt','bfgs','gd']
        for method in methods:
            x0 = np.array([2, 2])
            step_size = 0.0001
            max_itr = 20000
            step_tolerance = 10**-8
            obj_tolerance = 10**-10
            f = examples.rosenbrock(np.array([2.0, 2.0]), True)
            res = unconstrained.line_search(
                examples.rosenbrock,
                x0,
                step_size,
                obj_tolerance,
                step_tolerance,
                max_itr,
                method,
                init_step_len=0.005)
            X, Y = np.meshgrid(
                np.linspace(-5, 5),
                np.linspace(-5, 5),
                indexing='ij',
            )
            plot(
                X,
                Y,
                examples.rosenbrock,
                res['path_taken'],
                'rosenbrock min ' + method)
            plot_iterations_objective(res['path_taken'])
if __name__ == '__main__':
    unittest.main()
