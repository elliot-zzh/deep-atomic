import pytest
import logging
import numpy as np

from .utils import *

from deep_atomic import *


# TODO: refactor for more organized test code


def test_sum():
    # test full reduction
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = input.sum()
    res.backward()
    func = lambda x: sum(x)
    assert_close(numerical_grad(func, input), input.grad)


def test_add():
    # test tensor + scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: sum(x + a2)
    assert_close(numerical_grad(func, a1), a1.grad)

    # test tensor + tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: lambda y: sum(x + y)
    assert_close(numerical_grad(func(a2), a1), a1.grad)
    assert_close(numerical_grad(func(a1), a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: lambda y: sum(x + y)
    assert_close(numerical_grad(func(a2), a1), a1.grad)
    assert_close(numerical_grad(func(a1), a2), a2.grad)


def test_sub():
    # test tensor - scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 - a2).sum()
    res.backward()

    func = lambda x: sum(x - a2)
    assert_close(numerical_grad(func, a1), a1.grad)

    # test tensor - tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 - a2).sum()
    res.backward()

    func1 = lambda x: sum(x - a2)
    func2 = lambda x: sum(a1 - x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 - a2).sum()
    res.backward()

    func1 = lambda x: sum(x - a2)
    func2 = lambda x: sum(a1 - x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)


def test_mul():
    # test tensor * scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 * a2).sum()
    res.backward()

    func = lambda x: sum(x * a2)
    assert_close(numerical_grad(func, a1), a1.grad)

    # test tensor * tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 * a2).sum()
    res.backward()

    func1 = lambda x: sum(x * a2)
    func2 = lambda x: sum(a1 * x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 * a2).sum()
    res.backward()

    func1 = lambda x: sum(x * a2)
    func2 = lambda x: sum(a1 * x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)


def test_matmul():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4, 2)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 @ a2).sum()
    res.backward()

    func1 = lambda x: sum(x @ a2)
    func2 = lambda x: sum(a1 @ x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(5, 3, 4), np.random.rand(1, 4, 2)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 @ a2).sum()
    res.backward()

    func1 = lambda x: sum(x @ a2)
    func2 = lambda x: sum(a1 @ x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)


def test_div():
    # test tensor / scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 / a2).sum()
    res.backward()

    func = lambda x: sum(x / a2)
    assert_close(numerical_grad(func, a1), a1.grad)

    # test scalar / tensor
    a1_np, a2_np = np.random.rand(1), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np, requires_grad=False), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func = lambda x: sum(a1 / x)
    assert_close(numerical_grad(func, a2), a2.grad)

    # test tensor / tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func1 = lambda x: sum(x / a2)
    func2 = lambda x: sum(a1 / x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func1 = lambda x: sum(x / a2)
    func2 = lambda x: sum(a1 / x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)


def test_pow():
    # test tensor ** scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1**a2).sum()
    res.backward()

    func = lambda x: sum(x**a2)
    assert_close(numerical_grad(func, a1), a1.grad)

    # test scalar ** tensor
    a1_np, a2_np = np.random.rand(1), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np, requires_grad=False), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func = lambda x: sum(a1**x)
    assert_close(numerical_grad(func, a2), a2.grad)

    # test tensor ** tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func1 = lambda x: sum(x**a2)
    func2 = lambda x: sum(a1**x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func1 = lambda x: sum(x**a2)
    func2 = lambda x: sum(a1**x)

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)


def test_exp():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(exp(input))
    res.backward()

    func = lambda x: sum(exp(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_log():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(log(input))
    res.backward()

    func = lambda x: sum(log(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_sin():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(sin(input))
    res.backward()

    func = lambda x: sum(sin(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_cos():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(cos(input))
    res.backward()

    func = lambda x: sum(cos(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_tan():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(tan(input))
    res.backward()

    func = lambda x: sum(tan(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arcsin():
    input_np = np.clip(np.random.rand(3, 4) * 2 - 1, -0.99, 0.99)
    input = Tensor(input_np)
    res = sum(arcsin(input))
    res.backward()

    func = lambda x: sum(arcsin(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arccos():
    input_np = np.clip(np.random.rand(3, 4) * 2 - 1, -0.99, 0.99)
    input = Tensor(input_np)
    res = sum(arccos(input))
    res.backward()

    func = lambda x: sum(arccos(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arctan():
    input_np = np.random.rand(3, 4) * 10
    input = Tensor(input_np)
    res = sum(arctan(input))
    res.backward()

    func = lambda x: sum(arctan(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_sinh():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(sinh(input))
    res.backward()

    func = lambda x: sum(sinh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_cosh():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(cosh(input))
    res.backward()

    func = lambda x: sum(cosh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_tanh():
    input_np = np.random.rand(3, 4) * 2 - 1
    input = Tensor(input_np)
    res = sum(tanh(input))
    res.backward()

    func = lambda x: sum(tanh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arcsinh():
    input_np = np.random.rand(3, 4) * 10 - 5
    input = Tensor(input_np)
    res = sum(arcsinh(input))
    res.backward()

    func = lambda x: sum(arcsinh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arccosh():
    input_np = np.random.rand(3, 4) + 1.0
    input = Tensor(input_np)
    res = sum(arccosh(input))
    res.backward()

    func = lambda x: sum(arccosh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_arctanh():
    input_np = np.clip(np.random.rand(3, 4) * 1.8 - 0.9, -0.99, 0.99)
    input = Tensor(input_np)
    res = sum(arctanh(input))
    res.backward()

    func = lambda x: sum(arctanh(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_gradient_from_multiple_paths():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(input + input * input / exp(input))
    res.backward()

    func = lambda x: sum(x + x * x / exp(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_max():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(max(input, axis=-1))
    res.backward()

    func = lambda x: sum(max(x, axis=-1))
    assert_close(numerical_grad(func, input), input.grad)

    # test full reduction
    input_np = np.array([2.0, 2.0, 3.0, 3.0])
    input = Tensor(input_np)
    res = max(input)
    res.backward()
    expected_grad = np.array([0.0, 0.0, 0.5, 0.5])
    assert_close(expected_grad, input.grad)


def test_min():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(min(input, axis=-1))
    res.backward()

    func = lambda x: sum(min(x, axis=-1))
    assert_close(numerical_grad(func, input), input.grad)

    # test full reduction
    input_np = np.array([2.0, 2.0, 3.0, 3.0])
    input = Tensor(input_np)
    res = min(input)
    res.backward()
    expected_grad = np.array([0.5, 0.5, 0.0, 0.0])
    assert_close(expected_grad, input.grad)


def test_softmax():
    input_np = np.random.rand(3, 4)
    weight_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    weight = Tensor(weight_np)
    res = sum(
        softmax(input, axis=-1, temperature=0.6) * weight
    )  # must multiply a weight so that res != 3
    res.backward()

    func1 = lambda x: sum(softmax(x, axis=-1, temperature=0.6) * weight)
    assert_close(numerical_grad(func1, input), input.grad)
    func2 = lambda x: sum(softmax(input, axis=-1, temperature=0.6) * x)
    assert_close(numerical_grad(func2, weight), weight.grad)


def test_log_softmax():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(log_softmax(input, axis=-1))
    res.backward()

    func = lambda x: sum(log_softmax(x, axis=-1))
    assert_close(numerical_grad(func, input), input.grad)


def test_sigmoid():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(sigmoid(input))
    res.backward()

    func = lambda x: sum(sigmoid(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_silu():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(silu(input))
    res.backward()

    func = lambda x: sum(silu(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_gelu():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(gelu(input))
    res.backward()

    func = lambda x: sum(gelu(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_relu():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(relu(input))
    res.backward()

    func = lambda x: sum(relu(x))
    assert_close(numerical_grad(func, input), input.grad)


def test_reshape():
    input_np = np.random.rand(2, 6)
    input = Tensor(input_np)
    res = sum(softmax((input**2 + input).reshape(3, 4), axis=-1))
    res.backward()

    func = lambda x: sum(softmax((x**2 + x).reshape(3, 4)))
    assert_close(numerical_grad(func, input), input.grad)


def test_squeeze():
    input_np = np.random.rand(2, 1, 3)
    input = Tensor(input_np)
    res = ((input.squeeze(axis=1) ** 2) + input.squeeze(axis=1)).sum()
    res.backward()

    func = lambda x: sum((x.squeeze(axis=1) ** 2 + x.squeeze(axis=1)))
    assert_close(numerical_grad(func, input), input.grad)


def test_expand_dims():
    input_np = np.random.rand(2, 3)
    input = Tensor(input_np)
    res = ((input.expand_dims(axis=1) ** 2) + input.expand_dims(axis=1)).sum()
    res.backward()

    func = lambda x: sum((np.expand_dims(x, axis=1) ** 2 + np.expand_dims(x, axis=1)))
    assert_close(numerical_grad(func, input), input.grad)


def test_repeat_int_axis_none():
    input_np = np.random.rand(4)
    input = Tensor(input_np)
    res = ((input.repeat(2) ** 2) + input.repeat(2)).sum()
    res.backward()

    func = lambda x: ((x.repeat(2) ** 2) + x.repeat(2)).sum()
    assert_close(numerical_grad(func, input), input.grad)


def test_repeat_int_axis():
    input_np = np.random.rand(2, 3)
    input = Tensor(input_np)
    res = ((input.repeat(2, axis=0) ** 2) + input.repeat(2, axis=0)).sum()
    res.backward()

    func = lambda x: ((np.repeat(x, 2, axis=0) ** 2) + np.repeat(x, 2, axis=0)).sum()
    assert_close(numerical_grad(func, input), input.grad)


def test_repeat_sequence_axis_none():
    input_np = np.random.rand(3)
    input = Tensor(input_np)
    res = ((input.repeat([1, 2, 3]) ** 2) + input.repeat([1, 2, 3])).sum()
    res.backward()

    func = lambda x: ((np.repeat(x, [1, 2, 3]) ** 2) + np.repeat(x, [1, 2, 3])).sum()
    assert_close(numerical_grad(func, input), input.grad)


def test_repeat_sequence_axis():
    input_np = np.random.rand(2, 3)
    input = Tensor(input_np)
    res = ((input.repeat([2, 1], axis=0) ** 2) + input.repeat([2, 1], axis=0)).sum()
    res.backward()

    func = lambda x: (
        (np.repeat(x, [2, 1], axis=0) ** 2) + np.repeat(x, [2, 1], axis=0)
    ).sum()
    assert_close(numerical_grad(func, input), input.grad)


def test_tile():
    input_np = np.random.rand(2, 3)
    input = Tensor(input_np)
    res = ((input.tile(2, 2) ** 2) + input.tile(2, 2)).sum()
    res.backward()

    func = lambda x: ((tile(x, (2, 2)) ** 2) + tile(x, (2, 2))).sum()
    assert_close(numerical_grad(func, input), input.grad)


def test_where():
    condition = np.random.choice(np.array([True, False]), size=(3, 4))
    # test tensor and scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = where(condition, a1, a2).sum()
    res.backward()

    func = lambda x: sum(where(condition, x, a2))
    assert_close(numerical_grad(func, a1), a1.grad)

    # test tensor and tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = where(condition, a1, a2).sum()
    res.backward()

    func1 = lambda x: sum(where(condition, x, a2))
    func2 = lambda x: sum(where(condition, a1, x))

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = where(condition, a1, a2).sum()
    res.backward()

    func1 = lambda x: sum(where(condition, x, a2))
    func2 = lambda x: sum(where(condition, a1, x))

    assert_close(numerical_grad(func1, a1), a1.grad)
    assert_close(numerical_grad(func2, a2), a2.grad)
    

def test_topk():
    # largest
    input_np = np.random.rand(4, 5)
    input = Tensor(input_np)
    res = sum(topk(input, 2, axis=-1)[0])
    res.backward()
    
    func = lambda x: sum(topk(x, 2, axis=-1)[0])
    assert_close(numerical_grad(func, input), input.grad)
    
    # smallest
    input_np = np.random.rand(4, 5)
    input = Tensor(input_np)
    res = sum(topk(input, 2, axis=-1, largest=False)[0])
    res.backward()
    
    func = lambda x: sum(topk(x, 2, axis=-1, largest=False)[0])
    assert_close(numerical_grad(func, input), input.grad)