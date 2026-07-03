import numpy as np


class Tensor(np.ndarray):
    def __new__(
        subtype,
        arg0,  # arg0: ndarray | shape
        requires_grad=True,  # for backward
        dep=None,  # for backward
        dtype=np.float64,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        if isinstance(arg0, np.ndarray):
            obj = arg0.view(subtype)  # convert from ndarray
        else:
            obj = super().__new__(subtype, arg0, dtype, buffer, offset, strides, order)

        obj.requires_grad, obj.dep = requires_grad, dep
        obj.grad = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.requires_grad = getattr(obj, "requires_grad", False)
        self.dep = getattr(obj, "dep", None)

    # TODO: replace all to_np with view(np.ndarray)
    def to_np(self):
        return np.asarray(super())

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            # TODO: support Vector-Jacobian Product like pytorch
            assert np.size == 1  # must be scalar
            self.grad = np.array(1)
        else:
            self.grad = grad  # receive gradient from graph

        if self.dep is not None:
            self.dep.grad(self.grad)  # trigger graph

    # override operations
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # type conversion
        inputs_np = []
        requires_grad = self.requires_grad
        for input_ in inputs:
            if isinstance(input_, Tensor):
                inputs_np.append(input_.to_np())
                requires_grad = requires_grad or input_.requires_grad
            else:
                inputs_np.append(input_)

        result_np = super().__array_ufunc__(ufunc, method, *inputs_np, **kwargs)

        if not requires_grad:
            return Tensor(result_np, requires_grad=False)

        # if requires_grad, construct graph
        if ufunc is np.add:
            res = Tensor(result_np, dep=Add(*inputs))
        elif ufunc is np.subtract:
            res = Tensor(result_np, dep=Add(*inputs, sub=True))
        elif ufunc is np.multiply:
            res = Tensor(result_np, dep=Mul(*inputs))
        elif ufunc is np.divide:
            res = Tensor(result_np, dep=Mul(*inputs, div=True))
        elif ufunc is np.matmul:
            res = Tensor(result_np, dep=MatMul(*inputs))
        elif ufunc is np.exp:
            res = Tensor(result_np, dep=Exp(*inputs))
        elif ufunc is np.log:
            res = Tensor(result_np, dep=Log(*inputs))
        else:
            return NotImplemented

        return res

    def __pow__(
        self, other, mod=None
    ):  # TODO: more available types for other/exponent. currently non ndarray scalars
        base = self.to_np()
        # TODO: requires refactorization
        res = (base**other).view(Tensor)
        if self.requires_grad:
            res.requires_grad = True
            res.dep = Pow(self, other)
        return res


from .graph import *  # avoid looped dependencies
