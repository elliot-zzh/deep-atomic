import numpy as np
from .tensor import *
from .graph import *


def exp(input: Tensor):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.exp, "__call__", input)


def log(input: Tensor):
    # TODO: handle other methods for ufunc (or use torch like api?)
    return input.__array_ufunc__(np.log, "__call__", input)
