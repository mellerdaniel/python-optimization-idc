from numpy import linalg as LA
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def is_converged(change_in_f, step_len, obj_tol, param_tol):
    if change_in_f < obj_tol:
        return True
    if step_len < param_tol:
        return True
    return False


def gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter):
    x = x0
    previous_f_value = float('-inf')
    previous_x = float('-inf')
    f_values = []
    converged_flag = False
    for itr in range(max_iter):
        [current_f_value, grad, hessian] = f(x)
        f_values.append({'location':  x, 'val': current_f_value})
        step_len = LA.norm(x - previous_x)
        change_in_f = abs(previous_f_value - current_f_value)
        previous_x = x
        x = x - step_size * grad
        if is_converged(change_in_f, step_len, obj_tol, param_tol):
            converged_flag = True
            break
        previous_f_value = current_f_value

    return {'res': converged_flag, 'path_taken': f_values}

def find_step_len_wolfe_condition(x_k, p_k, c_1, alpha_0, f, f_grad, back_track_factor):
    is_objective = False
    new_alpha = alpha_0
    itr_cnt = 0
    while is_objective is False:
        [left_side, _, _] = f(x_k + new_alpha * p_k)
        [right_side, _, _] = f(x_k)
        right_side = right_side + c_1 * new_alpha * (f_grad @ p_k)
        # Updating alpha for the next round
        new_alpha = back_track_factor * new_alpha
        is_objective = (-0.001 < right_side - left_side)
        itr_cnt += 1
        if itr_cnt > 1000:
            return alpha_0
    return new_alpha

def line_search(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len=1.0,
                c_1=0.0001, back_track_factor=0.2):
    x = x0
    previous_f_value = float('-inf')
    previous_x = float('-inf')
    f_values = []
    converged_flag = False
    p_k = None
    previous_all_f_values = None
    b_k = None
    previous_pk = None
    b_k = b_k_1 = None
    for itr in range(max_iter):
        [current_f_value, grad, hessian] = f(x, True)
        f_values.append({'location':  x, 'val': current_f_value})
        step_len = LA.norm(x - previous_x)
        change_in_f = abs(previous_f_value - current_f_value)
        if itr == 0:
            new_alpha = init_step_len
        else:
            new_alpha = find_step_len_wolfe_condition(previous_pk, previous_pk, c_1, init_step_len, f, grad, back_track_factor)
        # defining the direction
        #Updating x_k+1
        if dir_selection_method == 'gd':
            p_k = - step_size * grad
        if dir_selection_method == 'nt':
            p_k = np.linalg.solve(hessian, -grad)
        if (dir_selection_method == 'bfgs'):
            if itr == 0:
                b_k_1 = np.linalg.inv(hessian)
                p_k = - b_k_1 @ grad
            else:
                s_k = x - previous_x
                y_k = grad - previous_all_f_values[1]
                b_k_1 = b_k - ((b_k @ (s_k.reshape(2, 1).dot(s_k.reshape(1, 2))) @ b_k) / (s_k.T @ b_k @ s_k)) + (y_k.reshape(2, 1).dot(y_k.reshape(1, 2)) / (y_k.T@s_k))
                p_k = - b_k_1 @ grad

        previous_x = x
        x = x + new_alpha * p_k
        if is_converged(change_in_f, step_len, obj_tol, param_tol):
            converged_flag = True
            break
        previous_f_value = current_f_value
        previous_all_f_values = [current_f_value, grad, hessian]
        b_k = b_k_1

        previous_pk = p_k

    return {'res': converged_flag, 'path_taken': f_values}



