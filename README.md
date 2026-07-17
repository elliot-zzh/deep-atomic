# Deep Atomic

A simple deep learning framework built on NumPy only. Mainly for practice and learning.

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

Most essential deep-learning operations are implemented. For those that also exist in NumPy, we follow NumPy's API conventions.

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

d = a < b                     # element-wise comparison, creates a boolean tensor
e = a <= b
c = a > b
c = a >= b
c = a == b
c = a != b
c = da.fmax(a, b)             # IMPORTANT: here da.fmax is identical to da.maximum, for simplicity. Same for da.fmin / da.minimum
c = da.maximum(a, b)
c = da.fmin(a, b)
c = da.minimum(a, b)

c = da.logical_and(d, e)                     # element-wise and
c = d & e                                    # equivalence
c = da.logical_or(d, e)                      # element-wise or
c = d | e
c = da.logical_xor(d, e)                     # element-wise xor
c = d ^ e
c = da.logical_not(d)                        # element-wise not
c = ~d
c = d.all(axis=-1, keepdims=False)           # logical AND reduction. axis=None, keepdims=False by default
c = d.any(axis=-1, keepdims=False)           # logical OR reduction. axis=None, keepdims=False by default
c = da.where(d, a, b)                        # returns elements chosen from a or b depending on condition

c = da.topk(a, 2, axis=-1, largest=True)    # same as pytorch. axis=-1, largest=True by default

c = da.sum(a)                               # shape: (1,)
c = da.sum(a, axis=1)                       # shape: (3,)
c = da.sum(a, axis=1, keepdims=True)        # shape: (3, 1)
# min, max, argmin, argmax follow the same signature
# all reductions set axis=None, keepdims=False by default

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

Autograd is supported via a computational graph.
_Currently only supports scalar source points._

```python
x = Tensor(np.random.rand(3, 4)) # requires_grad == True by default
res = ... # some calculation related to x. res is a **scalar** result
res.backward()
print(res.grad) # gradient computed!
```

### Neural Network Modules

Deep Atomic provides a PyTorch-style module system — `Module`, `Parameter`, `Buffer`, and the usual
traversal methods (`parameters`, `state_dict`, `train`/`eval`, etc.) all work the same way.

```python
# supported modules
from deep_atomic.nn import (
    Sequential, Linear, ReLU, Sigmoid, Tanh, SiLU, GELU, Softmax, LogSoftmax,
    ModuleList, ParameterList, BufferList,
    MSELoss, CrossEntropyLoss,
)

model = Sequential([
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
    Softmax(),
])

criterion = CrossEntropyLoss()   # MSELoss also available
logits = model(x)                # (batch, num_classes)
loss = criterion(logits, labels)
loss.backward()
```

### Optimizers

Optimizers basically follow PyTorch's API. Currently `SGD`, `Adam`/`AdamW`, and `Muon` are supported.

```python
from deep_atomic.optim import SGD, Adam, AdamW, Muon

opt_sgd  = SGD(model.parameters(),   lr=0.01, momentum=0.9)
opt_adam = Adam(model.parameters(),  lr=1e-3)
opt_adamw = AdamW(model.parameters(), lr=1e-3)
opt_muon = Muon(model.parameters(), lr=0.001)

opt = opt_adam
for epoch in range(epochs):
    for x_batch, y_batch in loader:
        opt.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        opt.step()
```



## To Do

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
- [x] basic neural network modules
- [x] optimizers and loss functions
- [ ] dataset pipelines
- [ ] save and load state dict file
- [ ] full training test
- [ ] benchmark with pytorch on CPU
- [ ] full LLM training
- [ ] finer type annotations, comments and documentation

## Development

We recommend managing dependencies with [uv](https://github.com/astral-sh/uv).
We use [pre-commit](https://github.com/pre-commit/pre-commit) to manage hooks that help lint and format our code.

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
