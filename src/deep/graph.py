import numpy as np
from .tensor import Tensor
from abc import ABC, abstractmethod


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
        # FIXME: handle operands that is not Tensor
        if self.a1 is not None:
            self.a1.backward(grad)
        if self.a2 is None:
            return
        if self.sub:
            self.a2.backward(-1 * grad)
        else:
            self.a2.backward(grad)


class Mul(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad * self.a2_np())
        if self.a2 is not None:
            self.a2.backward(self.a1_np() * grad)


class Div(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])

    def grad(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad * self.a2.to_np())
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
            self.a1.backward(grad @ self.a2.to_np().T)
        if self.a2 is not None:
            self.a2.backward(self.a1.to_np().T @ grad)


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
    def __init__(self, base: Tensor, exponent):
        self.base, self.exponent = base, exponent

    def grad(self, grad):
        self.input.backward(
            self.exponent * (self.base.to_np()) ** (self.exponent - 1) * grad
        )
