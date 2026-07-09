import numpy as np

from .utils import *


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
        elif isinstance(arg0, np.generic):
            obj = np.array(arg0).view(subtype)
        else:
            obj = super().__new__(subtype, arg0, dtype, buffer, offset, strides, order)

        obj.requires_grad, obj.dep = requires_grad, dep
        obj.depended_count = 0  # for topo sorting in backward
        return obj

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._requires_grad = value
        if value:
            # IMPORTANT: will reset gradient to zero
            self.grad = np.zeros(self.shape)
        else:
            self.grad = None

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # FIXME: ? when creating a view, we cannot rely on python's default reference mechanism. must share identical depended_count across views. should depended_count be a list?
        self.requires_grad = getattr(
            obj, "requires_grad", False
        )  # requires_grad not exist means creating a view from a non-Tensor object
        self.dep = getattr(obj, "dep", None)
        self.depended_count = getattr(obj, "depended_cound", 0)
        self.grad = getattr(obj, "grad", None)

    def to_np(self):
        return self.view(np.ndarray)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:  # source point of compute graph
            # TODO: support Vector-Jacobian Product like pytorch
            assert self.size == 1  # must be scalar
            self.grad += 1
        # otherwise receive gradient from graph
        elif grad.size == 1:
            self.grad += np.tile(grad, self.shape)
        elif is_broadcastable(grad.shape, self.shape):
            self.grad += grad
        elif grad.shape != self.shape:
            grad_ = grad
            dim_to_reduce = detect_broadcast_dim(self.shape, grad.shape)
            for dim, keepdims in dim_to_reduce:
                grad_ = grad_.sum(axis=dim, keepdims=keepdims)
            self.grad += grad_
        else:
            raise ValueError(
                f"Invalid gradient shape. Expected {self.shape}, but got {grad.shape}"
            )
        self.depended_count -= 1
        if self.dep is not None and self.depended_count <= 0:
            self.dep.backward(self.grad)  # trigger graph

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

        if "out" in kwargs and kwargs["out"] is not None:
            if not isinstance(kwargs["out"], tuple):
                kwargs["out"] = (kwargs["out"],)
            new_out = []
            for i in kwargs["out"]:
                # TODO: this will make kwargs['where'] ineffective when i is a tensor requiring grad! need to resolve this behavior
                if isinstance(i, Tensor):
                    if not i.requires_grad:
                        new_out.append(i.to_np())
                else:
                    new_out.append(i)
            if new_out:
                kwargs["out"] = tuple(new_out)
            else:
                del kwargs["out"]

        result_np = super().__array_ufunc__(ufunc, method, *inputs_np, **kwargs)

        if (
            ufunc
            in (
                np.equal,
                np.not_equal,
                np.greater,
                np.greater_equal,
                np.less,
                np.less_equal,
                np.logical_not,
                np.logical_and,
                np.logical_or,
                np.logical_xor,
            )
            or not requires_grad
        ):
            return Tensor(result_np, requires_grad=False)

        # if requires_grad, construct graph
        # TODO: implement gradient computation for other methods / ufuncs
        if ufunc is np.add:
            if method == "__call__":
                res = Tensor(result_np, dep=Add(*inputs))
            elif method == "reduce":
                axis = getattr(kwargs, "axis", None)
                keepdims = getattr(kwargs, "keepdims", False)
                res = Tensor(result_np, dep=Sum(*inputs, axis, keepdims))
            else:
                return NotImplemented
        elif ufunc is np.subtract:
            res = Tensor(result_np, dep=Add(*inputs, sub=True))
        elif ufunc is np.multiply:
            res = Tensor(result_np, dep=Mul(*inputs))
        elif ufunc is np.negative:
            res = Tensor(result_np, dep=Div(-1, *inputs))
        elif ufunc is np.divide:
            res = Tensor(result_np, dep=Div(*inputs))
        elif ufunc is np.matmul:
            res = Tensor(result_np, dep=MatMul(*inputs))
        elif ufunc is np.exp:
            res = Tensor(result_np, dep=Exp(*inputs))
        elif ufunc is np.exp2:
            res = Tensor(result_np, dep=Exp(Mul(*inputs, np.log(2))))
        elif ufunc is np.log:
            res = Tensor(result_np, dep=Log(*inputs))
        elif ufunc is np.log2:
            res = Tensor(result_np, dep=Div(Log(*inputs), np.log(2)))
        elif ufunc is np.log10:
            res = Tensor(result_np, dep=Div(Log(*inputs), np.log(10)))
        elif ufunc is np.pow:
            res = Tensor(result_np, dep=Pow(*inputs))
        elif ufunc is np.square:
            res = Tensor(result_np, dep=Pow(*inputs, 2))
        elif ufunc is np.abs:
            res = Tensor(result_np, dep=Abs(*inputs))
        elif ufunc is np.sin:
            res = Tensor(result_np, dep=Sin(*inputs))
        elif ufunc is np.cos:
            res = Tensor(result_np, dep=Cos(*inputs))
        elif ufunc is np.tan:
            res = Tensor(result_np, dep=Tan(*inputs, result_np))
        elif ufunc is np.arcsin:
            res = Tensor(result_np, dep=Arcsin(*inputs))
        elif ufunc is np.arccos:
            res = Tensor(result_np, dep=Arccos(*inputs))
        elif ufunc is np.arctan:
            res = Tensor(result_np, dep=Arctan(*inputs))
        elif ufunc is np.sinh:
            res = Tensor(result_np, dep=Sinh(*inputs, result_np))
        elif ufunc is np.cosh:
            res = Tensor(result_np, dep=Cosh(*inputs))
        elif ufunc is np.tanh:
            res = Tensor(result_np, dep=Tanh(*inputs, result_np))
        elif ufunc is np.arcsinh:
            res = Tensor(result_np, dep=Arcsinh(*inputs))
        elif ufunc is np.arccosh:
            res = Tensor(result_np, dep=Arccosh(*inputs))
        elif ufunc is np.arctanh:
            res = Tensor(result_np, dep=Arctanh(*inputs))
        else:
            return NotImplemented

        return res

    def reshape(self, *target_shape):
        res = Tensor(super().reshape(*target_shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            res.dep = Reshape(self, self.shape)
        return res

    def squeeze(self, axis):
        res = Tensor(super().squeeze(axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            res.dep = Squeeze(self, axis)
        return res

    def expand_dims(self, axis):
        res = Tensor(np.expand_dims(super(), axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            res.dep = ExpandDims(self, axis)
        return res

    def repeat(self, repeats, axis=None):
        res = Tensor(super().repeat(repeats, axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            res.dep = Repeat(self, repeats, axis)
        return res

    def tile(self, *reps):
        res = Tensor(np.tile(super(), reps), requires_grad=self.requires_grad)
        if self.requires_grad:
            res.dep = Tile(self, reps)
        return res
    
    def all(self, axis=None):
        return self.__array_ufunc__(np.logical_and, "reduce", self, axis=axis)
    
    def any(self, axis=None):
        return self.__array_ufunc__(np.logical_or, "reduce", self, axis=axis)


from .graph import *  # avoid looped dependencies
