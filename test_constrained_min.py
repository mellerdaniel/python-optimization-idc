import math as math
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from src.utils import *
import examples as examples
import src.contrained_min

def plot(X, Y, f, path_taken, title):
    params = list(zip(np.ravel(X), np.ravel(Y)))
    vals = []
    min = [0, 0]
    for p in params:
        v = f[0](np.array(p))[0]
        if v < min[0]:
            min = [v, p]
        vals.append(v)
    Z = np.array(vals).reshape(X.shape[0], X.shape[1])
    fig, ax = plt.subplots(figsize=(30, 30))

    ax.imshow(((-Y - X + 1 <= 0) &
               (Y - 1 <= 0) &
               (X - 2 <= 0) &
               (- Y <= 0)
               ).astype(int),
              extent=(-5, 5, -5, 5), origin='lower', alpha=0.3)
    CS = ax.contour(X, Y, Z)

    ax.clabel(CS, inline=True, fontsize=4)
    ax.set_title(title)
    for v in path_taken:
        plt.plot(v['location'][0], v['location'][1], 'x', c='green', ms=2)
    plt.show()


class TestConstrained(unittest.TestCase):

    def test_qp_min(self):
        path_taken = src.contrained_min.interior_pt(examples.test_qp, np.array(
            []), np.array([[0.1, 0.3, 0.7]]), np.array([[1]]), np.array([[0.4], [0.1], [0.5]]))

        plot_iterations_objective(path_taken=path_taken)

    def test_lp_min(self):
        path_taken = src.contrained_min.interior_pt_no_eq_constraint(
            examples.test_lp, np.array([[]]), np.array([]), np.array([]), np.array([[0.5], [0.75]]))

        d = np.linspace(-5, 5, 1000)
        X, Y = np.meshgrid(
            d,
            d,
            indexing='ij',
        )

        plot(
            X,
            Y,
            [examples.objective_2],
            path_taken,
            'lp minimization')
        plot_iterations_objective(path_taken=path_taken)

if __name__ == '__main__':
    unittest.main()
