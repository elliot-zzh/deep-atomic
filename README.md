# Deep Atomic

A simple deep learning framework built upon numpy only. Mainly for practice and learning.

## Usage

### Installation

```bash
pip install deep-atomic

# or using uv
uv add deep-atomic
```

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

c = a + b                     # addition
c = a - b                     # subtraction
c = a * b                     # element‑wise multiplication
c = a / b                     # element‑wise division
c = a ** b                    # element‑wise power
c = a @ b                     # matrix multiplication

c = da.exp(a)
c = da.log(a)
c = da.sin(a)
# the same for cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh

d = a < b                     # element-wise comparison, create a boolean tensor
e = a <= b
c = a > b
c = a >= b
c = a == b
c = a != b
c = da.fmax(a, b)             # IMPORTANT: here da.fmax is identical to da.maximum, for simplicity. same for da.fmin / da.minimum
c = da.maximum(a, b)
c = da.fmin(a, b)
c = da.minimum(a, b)

c = d & e                     # element-wise and
c = d | e                     # element-wise or
c = d ^ e                     # element-wise xor
c = da.logical_not(d)         # element-wise not
c = d.all(axis=-1)            # reduction of and operation
c = d.any(axis=-1)            # reduction of or operation
c = da.where(d, a, b)         # return elements chosen from a or b depending on condition

c = da.topk(a, 2, axis=-1, largest=True)    # same as pytorch. axis=-1, largest=True by default

c = da.sum(a)                               # shape: (1,)
c = da.sum(a, axis=1)                       # shape: (3,)
c = da.sum(a, axis=1, keepdims=True)        # shape: (3, 1)
# min, max, argmin, argmax follow the same signature

c = da.softmax(a, axis=-1, temperature=0.6) # support temperature. temperature=1 by default
c = da.log_softmax(a, axis=-1, temperature=0.6)

c = da.sigmoid(a)
c = da.silu(a)
c = da.relu(a)
c = da.gelu(a) # Deep Atomic uses the tanh approximation for speed and convenience

c = a.reshape(2, 6)
c = a.reshape(1, 12).squeeze(0)               # shape: (12,)
c = da.expand_dims(a, -1)                     # shape: (3, 4, 1)
c = a.expand_dims(-1)                         # method‑style alternative
c = a.repeat(2, axis=1)                       # shape: (3, 8)
c = da.tile(a, (2, 2))                        # shape: (6, 8)
c = a.tile(2, 2)                              # method‑style alternative
```

### Autograd

Autograd is supported via computational graph.
_Currently only support scalar source points._

```python
x = Tensor(np.random.rand(3, 4)) # requires_grad == True by default
res = ... # some calculation related to x. res is a **scalar** result
res.backward()
print(res.grad) # gradient get!
```

## Todo

- [ ] more basic operations and their autograd
    - [x] topk
    - [x] boolean operations
    - [x] gather or take_along_axis
    - [ ] scatter
    - [ ] convolution and 2d convolution
    - [ ] pooling
    - [ ] softmax attention
    - [ ] direct masking and indexing via `[]` syntax
    - [ ] einsum
    - [ ] normalization
- [ ] support backward with Vector-Jacobian Product like pytorch
- [ ] basic neural network modules
- [ ] optimizers and loss functions
- [ ] dataset pipelines
- [ ] save and load state dict file
- [ ] full training test
- [ ] benchmark with pytorch on CPU
- [ ] full LLM training
- [ ] finer type annotations, comments and documentation

## Development

Recommend manage dependencies using [uv](https://github.com/astral-sh/uv).
We use [pre-commit](https://github.com/pre-commit/pre-commit) to manage hooks which help to lint and format our code.

```bash
uv sync
pre-commit install # install pre-commit git hooks for lint and format
```

Run tests:

```bash
cd tests
pytest
```

Build wheels:

```bash
uv build
```
