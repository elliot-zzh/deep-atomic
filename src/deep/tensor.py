import numpy as np

class Tensor(np.ndarray):
    def __new__(subtype, arg0, # arg0: ndarray or shape
                requires_grad=True, dep=None, # for backward
                dtype=np.float64, buffer=None, offset=0,
                strides=None, order=None):
        if isinstance(arg0, np.ndarray):
            obj = arg0.view(subtype) # convert from ndarray
        else:
            obj = super().__new__(subtype, arg0, dtype,
                              buffer, offset, strides, order)
        
        obj.requires_grad, obj.dep = requires_grad, dep
        obj.grad = None
        
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.requires_grad = getattr(obj, 'requires_grad', True)
        self.dep = getattr(obj, 'dep', None)
        
    def to_np(self):
        return np.asarray(super())
                     
    def backward(self, grad=None):
        if not self.requires_grad: return
        if grad is None:
            assert np.size == 1 # must be scalar
            self.grad = np.array(1)
        else:
            self.grad = grad # receive gradient from graph
            
        if self.dep is not None:
            self.dep.grad(self.grad) # trigger graph
            
    # override operators
    # TODO: override with __array_ufunc__
    def __add__(self, other: 'Tensor'):
        a1_np = self.to_np()
        a2_np = other.to_np()
        # TODO: requires refactorization
        res = (a1_np + a2_np).view(Tensor)
        if self.requires_grad or other.requires_grad:
            res.requires_grad = True
            res.dep = Add(self, other)
        return res
    
    def __sub__(self, other: 'Tensor'):
        a1_np = self.to_np()
        a2_np = other.to_np()
        # TODO: requires refactorization
        res = (a1_np - a2_np).view(Tensor)
        if self.requires_grad or other.requires_grad:
            res.requires_grad = True
            res.dep = Add(self, other, sub=True)
        return res
    
    def __mul__(self, other: 'Tensor'):
        a1_np = self.to_np()
        a2_np = other.to_np()
        # TODO: requires refactorization
        res = (a1_np * a2_np).view(Tensor)
        if self.requires_grad or other.requires_grad:
            res.requires_grad = True
            res.dep = Mul(self, other)
        return res
    
    def __matmul__(self, other: 'Tensor'):
        a1_np = self.to_np()
        a2_np = other.to_np()
        # TODO: requires refactorization
        res = (a1_np @ a2_np).view(Tensor)
        if self.requires_grad or other.requires_grad:
            res.requires_grad = True
            res.dep = MatMul(self, other)
        return res
    
    def __truediv__(self, other):
        a1_np = self.to_np()
        a2_np = other.to_np()
        # TODO: requires refactorization
        res = (a1_np / a2_np).view(Tensor)
        if self.requires_grad or other.requires_grad:
            res.requires_grad = True
            res.dep = Mul(self, other, div=True)
        return res
    
    def __pow__(self, other, mod = None): # TODO: more available types for other/exponent. currently non ndarray scalars
        base = self.to_np()
        # TODO: requires refactorization
        res = (base ** other).view(Tensor)
        if self.requires_grad:
            res.requires_grad = True
            res.dep = Pow(self, other)
        return res
        

from .graph import * # avoid looped dependencies