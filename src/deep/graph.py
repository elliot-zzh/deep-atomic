import numpy as np
from .tensor import Tensor
from abc import ABC, abstractmethod

from .utils import *


class Op(ABC):
    @abstractmethod
    # TODO: Design: what is the type of grad? Tensor or np.ndarray
    def grad(self, grad):
        pass


class Add(Op):
    def __init__(self, a1, a2, sub=False):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])
        self.sub = sub

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad)
        if self.a2 is None:
            return
        if self.sub:
            grad = -grad
        self.a2.backward(grad)


class Mul(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad * self.a2_np)
        if self.a2 is not None:
            self.a2.backward(self.a1_np * grad)


class Div(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad / self.a2_np)
        if self.a2 is not None:
            self.a2.backward(-1 * self.a1.to_np() / (self.a2.to_np() ** 2) * grad)


class MatMul(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad @ self.a2.to_np().swapaxes(-1, -2))
        if self.a2 is not None:
            self.a2.backward(self.a1.to_np().swapaxes(-1, -2) @ grad)


class Exp(Op):
    def __init__(
        self, input: Tensor
    ):  # node of exp is introduced only when the input is a Tensor
        self.input = input

    def grad(self, grad):
        self.input.backward(np.exp(input.to_np()) * grad)


class Log(Op):
    def __init__(
        self, input: Tensor
    ):  # node of log is introduced only when the input is a Tensor
        self.input = input

    def grad(self, grad):
        self.input.backward(1 / input.to_np() * grad)


class Pow(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(self.a2_np * (self.a1_np) ** (self.a2_np - 1) * grad)
        if self.a2 is not None:
            self.a2.backward(np.log(self.a1_np) * (self.a1_np) ** (self.a2_np) * grad)


class Sum(Op):
    # TODO: follow numpy or torch?
    def __init__(self, input: Tensor, axis=None, keepdims=False):
        self.input = input
        self.axis = axis
        self.keepdims = keepdims

    def grad(self, grad):
        if self.keepdims or self.axis is None:
            self.input.backward(grad)
        else:
            self.input.backward(grad.expand_dims(self.axis))


class Abs(Op):
    def __init__(self, input: Tensor):
        self.input = input

    def grad(self, grad):
        grad = np.where(grad < 0, -grad, grad)
