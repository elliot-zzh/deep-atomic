import numpy as np

from .graph import *
from .tensor import *
from .utils import *


def add(x1, x2):
    return x1.__array_ufunc__(np.add, "__call__", x1, x2)


def sub(x1, x2):
    return x1.__array_ufunc__(np.subtract, "__call__", x1, x2)


def mul(x1, x2):
    return x1.__array_ufunc__(np.multiply, "__call__", x1, x2)


def div(x1, x2):
    return x1.__array_ufunc__(np.divide, "__call__", x1, x2)


def matmul(x1, x2):
    return x1.__array_ufunc__(np.matmul, "__call__", x1, x2)


def pow(x1, x2):
    return x1.__array_ufunc__(np.pow, "__call__", x1, x2)


def exp(x):
    return x.__array_ufunc__(np.exp, "__call__", x)


def log(x):
    return x.__array_ufunc__(np.log, "__call__", x)


# we do not implement exp2, log2, log10 etc. for that this framework is mainly for neural network training and for simplicity


def logical_not(x):
    return x.__array_ufunc__(np.logical_not, "__call__", x)


def logical_and(x1, x2):
    return x1.__array_ufunc__(np.logical_and, "__call__", x1, x2)


def logical_or(x1, x2):
    return x1.__array_ufunc__(np.logical_or, "__call__", x1, x2)


def logical_xor(x1, x2):
    return x1.__array_ufunc__(np.logical_xor, "__call__", x1, x2)


def all(x, axis=None, keepdims=False):
    return x.all(axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    return x.any(axis, keepdims=keepdims)


def sum(x: Tensor, axis=None, keepdims=False):
    return x.__array_ufunc__(np.add, "reduce", x, axis=axis, keepdims=keepdims)


def max(x: Tensor, axis=None, keepdims=False):
    res = Tensor(np.max(x.to_np(), axis=axis, keepdims=keepdims), requires_grad=False)
    if x.requires_grad:
        res.requires_grad = True
        if axis == None:
            res.dep = MinMax(x, None, keepdims, full_red_value=res.to_np())
        else:
            indices = np.argmax(x.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(x, axis, keepdims, indices=indices)
    return res


def min(x: Tensor, axis=None, keepdims=False):
    res = Tensor(np.min(x.to_np(), axis=axis, keepdims=keepdims), requires_grad=False)
    if x.requires_grad:
        res.requires_grad = True
        if axis == None:
            res.dep = MinMax(x, None, keepdims, full_red_value=res.to_np())
        else:
            indices = np.argmin(x.to_np(), axis=axis, keepdims=True)
            res.dep = MinMax(x, axis, keepdims, indices=indices)
    return res


def argmax(x: Tensor, axis=-1, keepdims=False):
    # no gradient concerned when performing argmax
    return Tensor(
        np.argmax(x.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )


def argmin(x: Tensor, axis=-1, keepdims=False):
    # no gradient concerned when performing armin
    return Tensor(
        np.argmin(x.to_np(), axis=axis, keepdims=keepdims), requires_grad=False
    )


def fmax(x1, x2):
    return x1.__array_ufunc__(np.fmax, "__call__", x1, x2)


def fmin(x1, x2):
    return x1.__array_ufunc__(np.fmin, "__call__", x1, x2)


# unlike pytorch, fmax and maximum, fmin and minimum are the same since Deep Atomic currently doesn't handle NaN
maximum = fmax
minimum = fmin


def softmax(x: Tensor, axis=-1, temperature=1):
    x /= temperature
    x = exp(x - max(x, axis=axis, keepdims=True))
    return x / sum(x, axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis=-1, temperature=1):
    x /= temperature
    x = x - max(x, axis=axis, keepdims=True)
    return x - log(sum(exp(x), axis=axis, keepdims=True))


def sigmoid(x: Tensor):
    pos = x.to_np() > 0
    x_np = x.to_np()
    res = np.zeros(x.shape)
    res[pos] = 1 / (1 + exp(-x_np[pos]))
    res[~pos] = exp(x_np[~pos]) / (1 + exp(x_np[~pos]))
    res = Tensor(res, requires_grad=x.requires_grad)
    if res.requires_grad:
        res.dep = Sigmoid(x, res.to_np())
    return res


def sin(x):
    return x.__array_ufunc__(np.sin, "__call__", x)


def cos(x):
    return x.__array_ufunc__(np.cos, "__call__", x)


def tan(x):
    return x.__array_ufunc__(np.tan, "__call__", x)


def arcsin(x):
    return x.__array_ufunc__(np.arcsin, "__call__", x)


def arccos(x):
    return x.__array_ufunc__(np.arccos, "__call__", x)


def arctan(x):
    return x.__array_ufunc__(np.arctan, "__call__", x)


def sinh(x):
    return x.__array_ufunc__(np.sinh, "__call__", x)


def cosh(x):
    return x.__array_ufunc__(np.cosh, "__call__", x)


def tanh(x):
    return x.__array_ufunc__(np.tanh, "__call__", x)


def arcsinh(x):
    return x.__array_ufunc__(np.arcsinh, "__call__", x)


def arccosh(x):
    return x.__array_ufunc__(np.arccosh, "__call__", x)


def arctanh(x):
    return x.__array_ufunc__(np.arctanh, "__call__", x)


def relu(x: Tensor):
    condition = x > 0
    return where(condition, x, 0)


def silu(x: Tensor):
    return x * sigmoid(x)


def gelu(x: Tensor):
    return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.44715 * x**3)))


def reshape(x: Tensor, target_shape):
    return x.reshape(*target_shape)


def squeeze(x: Tensor, axis):
    return x.squeeze(axis)


def expand_dims(x: Tensor, axis):
    return x.expand_dims(axis)


def repeat(x: Tensor, repeats, axis=None):
    return x.repeat(repeats, axis)


def tile(x: Tensor, reps):
    return x.tile(*reps)


def where(condition, x1, x2):
    requires_grad = False
    for i in (x1, x2):
        if isinstance(i, Tensor) and i.requires_grad:
            requires_grad = True
    res = Tensor(np.where(condition, x1, x2), requires_grad=requires_grad)
    if res.requires_grad:
        res.dep = Where(condition, x1, x2)
    return res


# TODO: API design. Should I follow pytorch and add a gather?
def take_along_axis(x: Tensor, indices, axis=-1):
    if isinstance(indices, Tensor):
        indices = indices.to_np()  # cut off gradient
    res = Tensor(np.take_along_axis(x, indices, axis), requires_grad=x.requires_grad)
    if res.requires_grad:
        res.dep = TakeAlongAxis(x, indices, axis)
    return res


# TODO: Tensor.implement scatter_


def topk(x: Tensor, kth, axis=-1, largest=True):
    if largest:
        indices = np.argpartition(x, x.shape[axis] - kth, axis)
        indices_idx = [slice(None)] * x.ndim
        indices_idx[axis] = slice(-kth, None)
    else:
        indices = np.argpartition(x, kth, axis)
        indices_idx = [slice(None)] * x.ndim
        indices_idx[axis] = slice(kth)
    indices = indices[tuple(indices_idx)]
    return take_along_axis(x, indices, axis), Tensor(indices, requires_grad=False)
