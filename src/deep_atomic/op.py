import numpy as np
from .tensor import *
from .graph import *


# implement add, sub, mul, div etc. here
def add(a1, a2):
    return a1.__array_ufunc__(np.add, "__call__", a1, a2)


def sub(a1, a2):
    return a1.__array_ufunc__(np.subtract, "__call__", a1, a2)


def mul(a1, a2):
    return a1.__array_ufunc__(np.multiply, "__call__", a1, a2)


def div(a1, a2):
    return a1.__array_ufunc__(np.divide, "__call__", a1, a2)


def matmul(a1, a2):
    return a1.__array_ufunc__(np.matmul, "__call__", a1, a2)


def pow(a1, a2):
    return a1.__array_ufunc__(np.pow, "__call__", a1, a2)


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
        res.requires_grad = True
        if axis == None:
            res.dep = MinMax(input, None, keepdims, full_red_value=res.to_np())
        else:
            indices = np.argmax(input.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(input, axis, keepdims, indices=indices)
    return res


def min(input: Tensor, axis=None, keepdims=False):
    res = Tensor(
        np.min(input.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )
    if input.requires_grad:
        res.requires_grad = True
        if axis == None:
            res.dep = MinMax(input, None, keepdims, full_red_value=res.to_np())
        else:
            indices = np.argmin(input.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(input, axis, keepdims, indices=indices)
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


def reshape(input: Tensor, target_shape):
    return input.reshape(*target_shape)


def squeeze(input: Tensor, axis):
    return input.squeeze(axis)


def expand_dims(input: Tensor, axis):
    return input.expand_dims(axis)


def repeats(input: Tensor, repeats, axis=None):
    return input.repeat(repeats, axis)


def tile(input: Tensor, reps):
    return input.tile(*reps)
