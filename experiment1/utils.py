import numpy as np
#import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exponential_function(x1, x2):
    return np.exp((-3 * x1) + x2)


def sin_function(x1, x2):
    return np.sin(x1*x2)


def poly_function_1(x1, x2):
    return (x1 * x2)/(x1**2 + x2)


def sic_function(x1, x2):
    return np.sinc(x1**2 + x2)


def create_data_splits():
    """
    function to return training/test
    data and labels depending on the
    experiment type
    """
    # experiment 1 involves modeling different types of functions (see True functions)

    true_functions = [('exp(-3x1 + x2)', exponential_function),
                      ('sin(xy)', sin_function),
                      ('xy/(x^2+y)', poly_function_1),
                      ('sinc(x^2+y)', sic_function)]
    features_range = [0, 1]  
    n_samples_train = 100  # training samples
    n_samples_test = 1000  # test samples
    n_dim = 2  # input dimension
    x = dict()  # training and test inputs

    x['train'] = np.random.uniform(features_range[0], features_range[1],
                                   size=n_samples_train * n_dim).reshape(n_samples_train, n_dim)  

    x['test'] = np.random.uniform(features_range[0], features_range[1],
                                  size=n_samples_test * n_dim).reshape(n_samples_test, n_dim)

    y = dict()  # training and test outputs

    for true_function in true_functions:
        y[true_function[0]] = compute_outputs(true_function[1], x)

    return x, y

def compute_outputs(func, inp_dict):
    """
    wrapper to call true functions
    """
    out = dict()
    out['train'] = func(inp_dict['train'][:, 0], inp_dict['train'][:, 1])
    out['test'] = func(inp_dict['test'][:, 0], inp_dict['test'][:, 1])
    return out

def compute_Rsquared(f_true, f_est):
    R2 = 1 - (np.mean((f_true - f_est)**2)/np.mean((f_true - np.mean(f_true))**2))
    return R2

def compute_mse_error(y_true, y_predicted):
    """
    function to compute mean square error
    between true outputs and predicted outputs
    """
    return np.mean((y_true - y_predicted) ** 2)


def compute_performance(x, y_true, best_tree):
    """
    compute R2 score and mean square error
    """
    arr1 = y_true.reshape((-1, 1))
    arr2 = best_tree.compute_outputs(x).reshape((-1, 1))

    r2_score = compute_Rsquared(arr1, arr2)
    mse = compute_mse_error(arr1, arr2)

    return r2_score, mse
