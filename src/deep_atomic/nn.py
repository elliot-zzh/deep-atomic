from .tensor import *
from .op import *
import numpy as np
from abc import ABC, abstractmethod

def init_param(*shape):
    return Tensor(np.random.randn(*shape))

class Module(ABC):
    def __init__(self):
        self.registered_parameters = []
        self.registerd_buffers = []
    
    def register_parameter(self, parameter: Tensor):
        self.registered_parameters.append(parameter)
        return parameter
        
    def register_buffer(self, buffer: Tensor):
        buffer.requires_grad = False
        self.registerd_buffers.append(buffer)
        return buffer
        
    def train(self):
        for i in self.registered_parameters:
            i.requires_grad = True
        for _, v in self.__dict__.items():
            if isinstance(v, Module):
                v.train()
    
    def eval(self):
        for i in self.registered_parameters:
            i.requires_grad = False
        for _, v in self.__dict__.items():
            if isinstance(v, Module):
                v.train()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
    
class Parameter(Module):
    def __init__(self, init_tensor: Tensor):
        super().__init__()
        self.weight = self.register_parameter(init_tensor)
        
    def forward(self): pass # dumb forward passing func

# TODO: implement ParameterList ParameterDict
  
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.weight = self.register_parameter(init_param(in_features, out_features))
        if bias: self.bias = self.register_parameter(init_param(out_features))
        else: self.bias = None
        
    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None: x += self.bias
        return x
    
class Sequential(Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
    
    # override .train and .eval since there exists a list of modules
    def train(self):
        for module in self.module_list:
            module.requires_grad = True
            
    def eval(self):
        for module in self.module_list:
            module.requires_grad = False
    
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
    
# add loss function layers