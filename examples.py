# Question 3 implememntation
import numpy as np
import math as math


def hessian_calc(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian
def quadratic_func(Q, x, calc_hessian):
    val = x.T.dot(Q.dot(x))
    grad = Q.dot(x) 
    hessian = np.empty([Q.shape[0], Q.shape[0]])
    if calc_hessian:
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[0]):
                hessian[i, j] = Q[i, j] + Q[j, i]
        # print(hessian.shape)
        # print(hessian)
    return [val, grad, hessian]


def test_qp(x, t):
    obj_val = t * (x[0][0] ** 2 + x[1][0] ** 2 + (x[2][0] + 1) ** 2) - math.log(x[0][0]) - math.log(x[1][0]) - math.log(x[2][0])
    grad = np.array([[2 * x[0][0] * t - (1 / x[0][0])], [2 * x[1][0] * t - (1 / x[1][0])], [2 * x[2][0] + 2 - (1 / x[2][0])]])
    hessian = np.array([[2 * t + (1 / (x[0][0] ** 2)), 0, 0], [0, 2 * t + (1 / (x[1][0] ** 2)), 0], [0, 0, 2 * t + (1 / (x[2][0] ** 2))]])
    return [obj_val, grad, hessian]


def test_lp(x, t):
    obj_val = -t * (x[0][0] + x[1][0]) - math.log(-(-x[0][0] - x[1][0] + 1)) - math.log(-(x[1][0] - 1)) - math.log(-(x[0][0] - 2)) - math.log(x[1][0])
    grad = np.array([[-1 - (1 / (x[0][0] + x[1][0] - 1 )) + (1 / (2 - x[0][0]))], [-1 - (1 / (x[0][0] + x[1][0] - 1)) + (1 / (1 - x[1][0])) - 1/x[1][0]]])
    hessian = np.array([[(1 / ((x[0][0] + x[1][0] - 1) ** 2)) + (1 / (2 - x[0][0]) ** 2), 1/ (x[0][0] + x[1][0] - 1)**2], [1 / (x[0][0] + x[1][0] - 1) ** 2, (1 / (x[0][0] + x[1][0] - 1 )) + (1/(1-x[1][0])**2) + 1/x[1][0] ** 2]])
    return [obj_val, grad, hessian]


def quadratic_func_1(x, calc_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    return quadratic_func(Q, x, calc_hessian)


def quadratic_func_2(x, calc_hessian=False):
    Q = np.array([[5, 0], [0, 1]])
    return quadratic_func(Q, x, calc_hessian)


def quadratic_func_3(x, calc_hessian=False):
    A = np.array([[math.sqrt(3) / 2, -0.5], [0.5, math.sqrt(3) / 2]])
    simple = np.array([[5, 0], [0, 1]])
    Q = A.T.dot(simple.dot(A))
    return quadratic_func(Q, x, calc_hessian)


def rosenbrock(x, calc_hessian=False):
    val = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    grad = np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1] - x[0] ** 2)])
    hessian = None
    if calc_hessian:
        hessian = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
    return [val, grad, hessian]
def abs_val(x):
    return [abs(x[0]) + abs(x[1]), 9]

def objective_2(x):
    return [0.5 * x[0] - x[1], 9]
