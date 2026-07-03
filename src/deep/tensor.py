import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, *args, requires_grad=True, dep=None, **kwargs):
        if args and isinstance(args[0], np.ndarray):
            dtype, order = kwargs.get('dtype', None), kwargs.get('order', 'C')
            return np.asarray(args[0], dtype=dtype, order=order).view(cls) # convert from ndarray
        else:
            return np.array(*args, **kwargs).view(cls)
        
    def __init__(self, *args, grad=True, dep=None, **kwargs):
        self.requires_grad = grad
        self.grad = None # gradient storage
        self.dep = None # dependency
        
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.requires_grad = getattr(obj, 'requires_grad', False)
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