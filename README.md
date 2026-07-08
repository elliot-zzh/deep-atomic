# deep-atomic

A simple deep learning framework built upon numpy only. Mainly for practice and learning.

## Usage

### Import

```python
import deep_atomic as da
import numpy as np   # required for tensor initialization and some operations
```

### Creating a Tensor

```python
# Create a tensor from a NumPy array (requires_grad=True by default)
a = da.Tensor(
    np.array([1, 2, 3], dtype=np.float64),
    requires_grad=True
)
```

### Supported Operations

Most essential deep‑learning operations are implemented. For those that also exist in NumPy, we follow NumPy's API conventions.

```python
a, b = da.Tensor(np.random.rand(3, 4)), da.Tensor(np.random.rand(3, 4))

# Arithmetic & math
c = a + b                     # addition
c = a - b                     # subtraction
c = a * b                     # element‑wise multiplication
c = a / b                     # element‑wise division
c = a ** b                    # element‑wise power
c = a @ b                     # matrix multiplication
c = da.exp(a)
c = da.log(a)

# Reductions
c = da.sum(a)                               # shape: (1,)
c = da.sum(a, axis=1)                       # shape: (3,)
c = da.sum(a, axis=1, keepdims=True)        # shape: (3, 1)
# min, max, argmin, argmax follow the same signature

# Softmax
c = da.softmax(a, axis=-1)
c = da.log_softmax(a, axis=-1)

# Shape manipulations
c = a.reshape(2, 6)
c = a.reshape(1, 12).squeeze(0)               # shape: (12,)
c = da.expand_dims(a, -1)                   # shape: (3, 4, 1)
c = a.expand_dims(-1)                         # method‑style alternative
c = a.repeat(2, axis=1)                       # shape: (3, 8)
c = da.tile(a, (2, 2))                      # shape: (6, 8)
c = a.tile(2, 2)                              # method‑style alternative
```

## Todo

- [ ] more basic operations and their autograd
    - [ ] convolution and 2d convolution
    - [ ] topk
    - [ ] boolean and masked operation
    - [ ] einsum
    - [ ] scatter
- [ ] basic neural network classes
- [ ] optimizers and loss functions
- [ ] full training test
- [ ] benchmark with pytorch on CPU
- [ ] attention layers and full LLM training
- [ ] finer type annotations, comments and documentation

## Installation

```bash
pip install deep_atomic

# or with uv
uv add deep_atomic
```

## Development

Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```
