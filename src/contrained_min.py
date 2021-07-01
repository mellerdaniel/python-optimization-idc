import numpy as np
import math as math

def lambda_stop_cond(p_nt, hessian):
    return math.sqrt(p_nt.T @ hessian @ p_nt)

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    m = ineq_constraints.shape[0] + eq_constraints_mat.shape[0]
    log_barrier_obj = m / t
    epsilon = 0.001
    internal_epsilon = 0.001
    internal_obj = float('inf')
    x = x0
    alpha = 0.01
    inner_cnt = 0
    outer_cnt = 0
    path_taken = []
    while log_barrier_obj > epsilon:
        while internal_obj > internal_epsilon:
            [val, gradient, hessian] = func(x, t)
            top_lhs = np.concatenate((hessian, eq_constraints_mat.T), axis=1)
            bottom_lhs = np.concatenate((eq_constraints_mat, np.array([[0]])), axis=1)
            lhs = np.concatenate((top_lhs, bottom_lhs))
            rhs = np.concatenate((-gradient, np.array([[0]])))
            solutions = np.linalg.solve(lhs, rhs )
            p_nt = solutions[:3, :]
            path_taken.append({'location':  x, 'val': val})
            x = x + alpha * p_nt
            internal_obj = 0.5 * (lambda_stop_cond(p_nt, hessian))**2
            inner_cnt += 1
        t = t * mu
        log_barrier_obj = m / t
        outer_cnt += 1
    return path_taken




def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    m = ineq_constraints.shape[0] + eq_constraints_mat.shape[0]
    log_barrier_obj = m / t
    epsilon = 0.001
    internal_epsilon = 0.001
    internal_obj = float('inf')
    x = x0
    alpha = 0.01
    inner_cnt = 0
    outer_cnt = 0
    path_taken = []
    while log_barrier_obj > epsilon:
        while internal_obj > internal_epsilon:
            [val, gradient, hessian] = func(x, t)
            top_lhs = np.concatenate((hessian, eq_constraints_mat.T), axis=1)
            bottom_lhs = np.concatenate((eq_constraints_mat, np.array([[0]])), axis=1)
            lhs = np.concatenate((top_lhs, bottom_lhs))
            rhs = np.concatenate((-gradient, np.array([[0]])))
            solutions = np.linalg.solve(lhs, rhs )
            p_nt = solutions[:3, :]
            path_taken.append({'location':  x, 'val': val})
            x = x + alpha * p_nt
            internal_obj = 0.5 * (lambda_stop_cond(p_nt, hessian))**2
            inner_cnt += 1
        t = t * mu
        log_barrier_obj = m / t
        outer_cnt += 1
    return path_taken

def interior_pt_no_eq_constraint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    m = ineq_constraints.shape[0] + eq_constraints_mat.shape[0]
    log_barrier_obj = m / t
    epsilon = 0.0001
    internal_epsilon = 0.00001
    internal_obj = float('inf')
    x = x0
    alpha = 0.01
    inner_cnt = 0
    outer_cnt = 0
    previous_x = float('inf')
    previous_val = float('inf')
    path_taken = []
    while log_barrier_obj > epsilon:
        while internal_obj > internal_epsilon:
            [val, gradient, hessian] = func(x, t)
            p_nt = np.linalg.solve(hessian, -gradient)
            path_taken.append({'location':  x, 'val': val})
            previous_x = x
            x = x + alpha * p_nt
            internal_obj = 0.5 * (lambda_stop_cond(p_nt, hessian))**2
            change_in_x = np.linalg.norm(x - previous_x)
            change_in_val = abs(previous_val - val)
            print('inner loop cnt: %d and obj: %f, val: %f, change in x: %f, change in val %f' % (inner_cnt, internal_obj, val, change_in_x, change_in_val))
            previous_val = val
            inner_cnt += 1
        t = t * mu
        log_barrier_obj = m / t
        outer_cnt += 1
    return path_taken