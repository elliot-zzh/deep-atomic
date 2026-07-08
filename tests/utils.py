import numpy as np

from deep_atomic import *


def assert_close(expected, actual, rtol=5e-05, atol=1e-08):
    delta = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    if np.isscalar(delta):
        assert delta <= tol
    else:
        assert (delta <= tol).all()


# TODO: more to implement on this utility. handle multi-input func, handle vector-valued output
def numerical_grad(func, input_, eps=1e-5):
    original_shape = input_.shape
    input_ = input_.reshape(-1)

    X_plus = np.tile(input_, (input_.size, 1))  # Copy x0 into n rows
    np.fill_diagonal(X_plus, X_plus.diagonal() + eps)
    X_plus = X_plus.reshape([input_.size] + list(original_shape))

    X_minus = np.tile(input_, (input_.size, 1))
    np.fill_diagonal(X_minus, X_minus.diagonal() - eps)
    X_minus = X_minus.reshape([input_.size] + list(original_shape))

    res = np.zeros([input_.size])
    # use a for loop to iterate through all parameters
    # slower but ensure correct and more convenient operations in func
    # e.g. .sum(axis=None) appeared in func will cause reduction of the added parameter dim, which is undesired
    for i in range(input_.size):
        # central difference
        res[i] = (func(X_plus[i]) - func(X_minus[i])) / (eps * 2)

    return res.reshape(original_shape)
