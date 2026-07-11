from .tensor import *
from .op import *
import numpy as np
from abc import ABC, abstractmethod

def init_param(*shape):
    return Tensor(np.random.randn(*shape))

class Parameter(Tensor): pass

class Buffer(Tensor):
    def __new__(
        subtype,
        arg0,  # arg0: ndarray | shape
        requires_grad=False,  # buffers are not trained
        dep=None,  # for backward
        dtype=np.float64,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        return super().__new__(
            subtype,
            arg0,
            requires_grad=False,
            dep=None,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )

class Module(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
        
    def __init__(self):
        # TODO: use OrderedDict?
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
        
    def state_dict(self, prefix=''):
        state_dict_ = {}
        if prefix: prefix += '.'
        for name, paramter in self._parameters.items():
            state_dict_[prefix + name] = paramter
        for name, buffer in self._buffers.items():
            state_dict_[prefix + name] = buffer
        for name, module in self._modules.items():
            state_dict_ = {**state_dict_, **module.state_dict(prefix=name)}
        return state_dict_
                
    def modules(self, recurse=True):
        for _name, module in self.named_modules(recurse=recurse):
            yield module
            
    def named_module(self, prefix='', remove_duplicate=True, memo = None):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            if prefix: prefix += '.'
            for name, module in self._modules.items():
                yield from module.named_modules(prefix=prefix + name, remove_duplicate=remove_duplicate, memo=memo)
                
    def _named_members(self, get_members_func, prefix='', recurse=True, remove_duplicate=True):
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else ((prefix, self),)
        )
        for module_prefix, module in modules:
            members = get_members_func(module)
            if module_prefix: module_prefix += '.'
            for k, v in members:
                if v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                yield module_prefix + k, v
                
    def parameters(self, recurse=True):
        for _name, parameter in self.named_parameters(recurse=recurse):
            yield parameter
    
    def named_parameters(self, recurse=True, prefix='', remove_duplicate=True):
        gen = self._named_members(
            get_members_func=lambda x: x._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen
        
    def buffers(self, recurse=True):
        for _name, buffer in self.named_buffers(recurse=recurse):
            yield buffer
    
    def named_buffers(self, recurse=True, prefix='', remove_duplicate=True):
        gen = self._named_members(
            get_members_func=lambda x: x._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen
    
    def train(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def eval(self):
        for param in self.parameters():
            param.requires_grad = False


class ParameterList(Module):
    def __init__(self, values: list[Tensor]):
        super().__init__()
        self._size = len(values)
        for idx, value in enumerate(values):
            self[idx] = value
    
    # TODO: edit implementation to make it same as any python list
    def _get_abs_index_string(self, idx: int):
        if idx < 0: idx += len(self)
        return str(idx)

    def __setitem__(self, idx: int, value: Tensor):
        if not isinstance(value, Parameter): value = value.view(Parameter)
        idx = self._get_abs_index_string(idx)
        return setattr(self, idx, value)
    
    def __len__(self): return self._size
    
    def __getitem__(self, idx: int):
        # TODO: support slice
        idx = self._get_abs_index_string(idx)
        return getattr(self, idx, None)
    
    def __iter__(self):
        return iter(self[i] for i in range(len(self)))
    
    def forward(self): pass
    
    
class ModuleList(Module):
    def __init__(self, values: list[Module]):
        super().__init__()
        self._size = len(values)
        for idx, value in enumerate(values):
            self[idx] = value
    
    # TODO: edit implementation to make it same as any python list
    def _get_abs_index_string(self, idx: int):
        if idx < 0: idx += len(self)
        return str(idx)

    def __setitem__(self, idx: int, value: Module):
        idx = self._get_abs_index_string(idx)
        return setattr(self, idx, value)
    
    def __len__(self): return self._size
    
    def __getitem__(self, idx: int):
        # TODO: support slice
        idx = self._get_abs_index_string(idx)
        return getattr(self, idx, None)
    
    def __iter__(self):
        return iter(self[i] for i in range(len(self)))
    
    def forward(self): pass


# TODO: implement BufferList, ParameterDict, BufferDict, ModuleDict

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.weight = Parameter(init_param(in_features, out_features))
        if bias: self.bias = Parameter(init_param(out_features))
        else: self.bias = None
        
    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None: x += self.bias
        return x
    
class Sequential(Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = ModuleList(module_list)
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return sigmoid(x)
    
class Tanh(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return tanh(x)
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return relu(x)
    
class SiLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return silu(x)
    
class GELU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return gelu(x)
    
class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        
    def forward(self, x):
        return softmax(x, self.axis)
    
class LogSoftmax(Module):
    def __init__(self,axis=-1):
        super().__init__()
        self.axis = axis
        
    def forward(self, x):
        return log_softmax(x, self.axis)
    
# TODO: add loss function layers