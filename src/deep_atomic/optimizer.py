from .tensor import *
from .op import *
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass