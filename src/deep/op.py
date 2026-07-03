import numpy as np
from .tensor import *
from .graph import *

def exp(input: Tensor):
    input_np = input.to_np()
    # TODO: requires refactorization
    res = np.exp(input_np).view(Tensor)
    if input.requires_grad:
        res.requires_grad = True
        res.dep = Exp(input)
    return res

def log(input: Tensor):
    input_np = input.to_np()
    # TODO: requires refactorization
    res = np.log(input_np).view(Tensor)
    if input.requires_grad:
        res.requires_grad = True
        res.dep = Log(input)
    return res