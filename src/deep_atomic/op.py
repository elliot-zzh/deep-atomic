import numpy as np

from .graph import *
from .tensor import *
from .utils import *


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
    return input.__array_ufunc__(np.exp, "__call__", input)


def log(input):
    return input.__array_ufunc__(np.log, "__call__", input)


# we do not implement exp2, log2, log10 etc. for that this framework is mainly for neural network training and for simplicity


def logical_not(input):
    return input.__array_ufunc(np.logical_not, "__call__", input)


def all(input, axis=None):
    return input.all(axis)


def any(input, axis=None):
    return input.any(axis)


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


def fmax(a1, a2):
    return a1.__array_ufunc__(np.fmax, "__call__", a1, a2)


def fmin(a1, a2):
    return a1.__array_ufunc__(np.fmin, "__call__", a1, a2)


# unlike pytorch, fmax and maximum, fmin and minimum are the same since Deep Atomic currently doesn't handle NaN
maximum = fmax
minimum = fmin


def softmax(input: Tensor, axis=-1, temperature=1):
    input /= temperature
    input = exp(input - max(input, axis=axis, keepdims=True))
    return input / sum(input, axis=axis, keepdims=True)


def log_softmax(input: Tensor, axis=-1, temperature=1):
    input /= temperature
    input = input - max(input, axis=axis, keepdims=True)
    return input - log(sum(exp(input), axis=axis, keepdims=True))


def sigmoid(input: Tensor):
    res = 1 / (1 + exp(-input))
    if res.requires_grad:
        res.dep = Sigmoid(input, res.to_np())
    return res


def sin(input):
    return input.__array_ufunc__(np.sin, "__call__", input)


def cos(input):
    return input.__array_ufunc__(np.cos, "__call__", input)


def tan(input):
    return input.__array_ufunc__(np.tan, "__call__", input)


def arcsin(input):
    return input.__array_ufunc__(np.arcsin, "__call__", input)


def arccos(input):
    return input.__array_ufunc__(np.arccos, "__call__", input)


def arctan(input):
    return input.__array_ufunc__(np.arctan, "__call__", input)


def sinh(input):
    return input.__array_ufunc__(np.sinh, "__call__", input)


def cosh(input):
    return input.__array_ufunc__(np.cosh, "__call__", input)


def tanh(input):
    return input.__array_ufunc__(np.tanh, "__call__", input)


def arcsinh(input):
    return input.__array_ufunc__(np.arcsinh, "__call__", input)


def arccosh(input):
    return input.__array_ufunc__(np.arccosh, "__call__", input)


def arctanh(input):
    return input.__array_ufunc__(np.arctanh, "__call__", input)


def relu(input: Tensor):
    condition = input > 0
    return where(condition, input, 0)


def silu(input: Tensor):
    return input * sigmoid(input)


def gelu(input: Tensor):
    return 0.5 * input * (1 + tanh(0.7978845608028654 * (input + 0.44715 * input**3)))


def reshape(input: Tensor, target_shape):
    return input.reshape(*target_shape)


def squeeze(input: Tensor, axis):
    return input.squeeze(axis)


def expand_dims(input: Tensor, axis):
    return input.expand_dims(axis)


def repeat(input: Tensor, repeats, axis=None):
    return input.repeat(repeats, axis)


def tile(input: Tensor, reps):
    return input.tile(*reps)


def where(condition, a1, a2):
    requires_grad = False
    for i in (a1, a2):
        if isinstance(i, Tensor) and i.requires_grad:
            requires_grad = True
    res = Tensor(np.where(condition, a1, a2), requires_grad=requires_grad)
    if res.requires_grad:
        res.dep = Where(condition, a1, a2)
    return res


# TODO: API design. Should I follow pytorch and add a gather?
def take_along_axis(input: Tensor, indices, axis=-1):
    if isinstance(indices, Tensor):
        indices = indices.to_np()  # cut off gradient
    res = Tensor(
        np.take_along_axis(input, indices, axis), requires_grad=input.requires_grad
    )
    if res.requires_grad:
        res.dep = TakeAlongAxis(input, indices, axis)
    return res


# TODO: Tensor.implement scatter_


def topk(input: Tensor, kth, axis=-1, largest=True):
    if largest:
        indices = np.argpartition(input, input.shape[axis] - kth, axis)
        indices_idx = [slice(None)] * input.ndim
        indices_idx[axis] = slice(-kth, None)
    else:
        indices = np.argpartition(input, kth, axis)
        indices_idx = [slice(None)] * input.ndim
        indices_idx[axis] = slice(kth)
    indices = indices[tuple(indices_idx)]
    return take_along_axis(input, indices, axis), Tensor(indices, requires_grad=False)
