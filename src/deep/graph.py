import numpy as np
from .tensor import Tensor
from abc import ABC, abstractmethod

from .utils import *


class Op(ABC):
    @abstractmethod
    def backward(self, grad):
        pass


class SingleOp(Op):
    def __init__(self, input: Tensor):
        self.input = input
        self.input.depended_count += 1


class TwoOp(Op):
    def __init__(self, a1, a2):
        self.a1_np, self.a2_np = (
            i.to_np() if isinstance(i, Tensor) else i for i in [a1, a2]
        )
        self.a1, self.a2 = (i if isinstance(i, Tensor) else None for i in [a1, a2])
        if isinstance(self.a1, Tensor):
            self.a1.depended_count += 1
        if isinstance(self.a2, Tensor):
            self.a2.depended_count += 1


class Add(TwoOp):
    def __init__(self, a1, a2, sub=False):
        super().__init__(a1, a2)
        self.sub = sub

    def backward(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad)
        if self.a2 is None:
            return
        if self.sub:
            grad = -grad
        self.a2.backward(grad)


class Mul(TwoOp):
    def __init__(self, a1, a2):
        super().__init__(a1, a2)

    def backward(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad * self.a2_np)
        if self.a2 is not None:
            self.a2.backward(self.a1_np * grad)


class Div(TwoOp):
    def __init__(self, a1, a2):
        super().__init__(a1, a2)

    def backward(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad / self.a2_np)
        if self.a2 is not None:
            self.a2.backward(-1 * self.a1_np / (self.a2_np**2) * grad)


class MatMul(TwoOp):
    def __init__(self, a1, a2):
        super().__init__(a1, a2)

    def backward(self, grad):
        if self.a1 is not None:
            self.a1.backward(grad @ self.a2_np.swapaxes(-1, -2))
        if self.a2 is not None:
            self.a2.backward(self.a1_np.swapaxes(-1, -2) @ grad)


class Exp(SingleOp):
    def __init__(
        self, input: Tensor
    ):  # node of exp is introduced only when the input is a Tensor
        super().__init__(input)

    def backward(self, grad):
        self.input.backward(np.exp(self.input.to_np()) * grad)


class Log(SingleOp):
    def __init__(
        self, input: Tensor
    ):  # node of log is introduced only when the input is a Tensor
        super().__init__(input)

    def backward(self, grad):
        self.input.backward(1 / self.input.to_np() * grad)


class Pow(TwoOp):
    def __init__(self, a1, a2):
        super().__init__(a1, a2)

    def backward(self, grad):
        if self.a1 is not None:
            self.a1.backward(self.a2_np * (self.a1_np) ** (self.a2_np - 1) * grad)
        if self.a2 is not None:
            self.a2.backward(np.log(self.a1_np) * (self.a1_np) ** (self.a2_np) * grad)


class Sum(SingleOp):
    def __init__(self, input: Tensor, axis, keepdims):
        super().__init__(input)
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad):
        if self.axis is None or self.keepdims:
            self.input.backward(grad)
        else:
            self.input.backward(np.expand_dims(grad, self.axis))


class Abs(SingleOp):
    def __init__(self, input: Tensor):
        super().__init__(input)

    def backward(self, grad):
        self.input.backward(np.where(grad < 0, -grad, grad))


class MinMax(SingleOp):
    def __init__(
        self, input: Tensor, axis, keepdims, indices=None, full_red_value=None
    ):
        super().__init__(input)
        self.indices = indices  # result of argmin / argmax. when axis is not None
        self.axis, self.keepdims = axis, keepdims
        self.full_red_value = full_red_value  # for faster compute when axis is None
        if axis is None:
            assert full_red_value is not None
        else:
            assert indices is not None

    def backward(self, grad):
        if self.axis is None:
            if not self.keepdims:
                grad = np.tile(grad, self.input.shape)
            mask = self.input.to_np() == self.full_red_value
            grad /= np.count_nonzero(
                mask
            )  # distribute gradient evenly across max values. following pytorch implementation
        else:
            if not self.keepdims:
                tile_reps = [1] * self.input.ndim
                tile_reps[self.axis] = self.input.shape[self.axis]
                grad = np.expand_dims(grad, axis=self.axis)
                grad = np.tile(grad, tile_reps)
            baseline_expansion_axis = list(range(self.input.ndim))
            baseline_expansion_axis.pop(self.axis)
            baseline_indices = np.expand_dims(
                np.arange(self.input.shape[self.axis]), axis=baseline_expansion_axis
            )
            mask = self.indices == baseline_indices
        self.input.backward(np.where(mask, grad, 0.0))


class Reshape(SingleOp):
    def __init__(self, input: Tensor, from_):
        super().__init__(input)
        self.from_ = from_

    def backward(self, grad):
        self.input.backward(grad.reshape(self.from_))


class Squeeze(SingleOp):
    def __init__(self, input: Tensor, axis):
        super().__init__(input)
        self.axis = axis

    def backward(self, grad):
        self.input.backward(np.expand_dims(grad, self.axis))


class ExpandDims(SingleOp):
    def __init__(self, input: Tensor, axis):
        super().__init__(input)
        self.axis = axis

    def backward(self, grad):
        self.input.backward(grad.squeeze(self.axis))
