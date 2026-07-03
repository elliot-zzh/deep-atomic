import numpy as np

from deep import *


def assert_close(baseline, eval, eps=1e-10):
    assert np.logical_and(baseline - eps < eval, eval < baseline + eps).all()


def numerical_grad(func, input_, eps=1e-5, use_log=True):
    original_shape = input_.shape
    if isinstance(input_, Tensor):
        input_.requires_grad = False
    input_ = input_.reshape(-1)

    X_plus = np.tile(input_, (input_.size, 1))  # Copy x0 into n rows
    np.fill_diagonal(X_plus, X_plus.diagonal() + eps)
    X_plus = X_plus.reshape([input_.size] + list(original_shape))

    X_minus = np.tile(input_, (input_.size, 1))
    np.fill_diagonal(X_minus, X_minus.diagonal() - eps)
    X_minus = X_minus.reshape([input_.size] + list(original_shape))

    delta = func(X_plus) - func(X_minus)

    # central difference
    if use_log:
        res = (delta / (np.abs(delta) + 1e-16)) * np.exp(
            np.log(np.abs(delta)) - np.log(eps * 2)
        )
    else:
        res = delta / (eps * 2)

    return res.reshape(original_shape)
