import numpy as np
from .tensor import *
from .graph import *

# TODO: should we implement add, sub, mul, div etc. here? (or just use np.add etc. directly)


def exp(input):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.exp, "__call__", input)


def log(input):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.log, "__call__", input)


# we do not implement exp2, log2, log10 etc. for that this framework is mainly for neural network training and for simplicity


def sum(input: Tensor, axis=None, keepdims=False):
    return input.__array_ufunc__(np.add, "reduce", input, axis=axis, keepdims=keepdims)


def max(input: Tensor, axis=None, keepdims=False):
    res = Tensor(
        np.max(input.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )
    if input.requires_grad:
        # FIXME: wrap the following two lines to ensure safety modification of requires_grad
        res.requires_grad = True
        res.grad = np.zeros(res.shape)
        if axis == None:
            # TODO: distribute gradient flow evenly
            pass
        else:
            indices = np.argmax(input.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(input, indices, axis, keepdims)
    return res


def min(input: Tensor, axis=None, keepdims=False):
    res = Tensor(
        np.min(input.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )
    if input.requires_grad:
        # FIXME: wrap the following two lines to ensure safety modification of requires_grad
        res.requires_grad = True
        res.grad = np.zeros(res.shape)
        if axis == None:
            # TODO: distribute gradient flow evenly
            pass
        else:
            indices = np.argmin(input.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(input, indices, axis, keepdims)
    return res


def argmax(input: Tensor, axis=-1, keepdims=False):
    # no gradient concerned when performing argmax
    return Tensor(
        np.argmax(input.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )


def argmin(input: Tensor, axis=-1, keepdims=False):
    # no gradient concerned when performing armin
    return Tensor(
        np.argmin(input.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )


def softmax(input: Tensor, axis=-1):
    input = exp(input - max(input, axis=axis, keepdims=True))
    return input / sum(input, axis=axis, keepdims=True)


def log_softmax(input: Tensor, axis=-1):
    input = input - max(input, axis=axis, keepdims=True)
    return input - log(sum(exp(input), axis=axis, keepdims=True))
