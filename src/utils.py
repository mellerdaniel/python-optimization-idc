import math as math
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np

def get_val(x):
    return x["val"]

def plot_iterations_objective(path_taken):
    x_axis = range(len(path_taken))
    values = list(map(get_val, path_taken))
    plt.plot(x_axis, values, 'ro', ms=2)
    plt.xlabel('iterations')
    plt.ylabel('objective function')
    plt.show()
