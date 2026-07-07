import pytest
import logging
import numpy as np

from .utils import *

from deep import *


def test_sum():
    # test full reduction
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = input.sum()
    res.backward()
    func = lambda x: sum(sum(x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_add():
    # test tensor + scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: sum(sum(x + a2, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a1).to_np(), a1.grad)

    # test tensor + tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: lambda y: sum(sum(x + y, axis=-1), axis=-1)
    assert_close(numerical_grad(func(a2), a1).to_np(), a1.grad)
    assert_close(numerical_grad(func(a1), a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 + a2).sum()
    res.backward()

    func = lambda x: lambda y: sum(sum(x + y, axis=-1), axis=-1)
    assert_close(numerical_grad(func(a2), a1).to_np(), a1.grad)
    assert_close(numerical_grad(func(a1), a2).to_np(), a2.grad)


def test_sub():
    # test tensor - scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 - a2).sum()
    res.backward()

    func = lambda x: sum(sum(x - a2, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a1).to_np(), a1.grad)

    # test tensor - tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 - a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x - a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 - x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 - a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x - a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 - x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


def test_mul():
    # test tensor * scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 * a2).sum()
    res.backward()

    func = lambda x: sum(sum(x * a2, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a1).to_np(), a1.grad)

    # test tensor * tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 * a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x * a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 * x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 * a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x * a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 * x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


def test_matmul():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4, 2)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 @ a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x @ a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 @ x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(5, 3, 4), np.random.rand(1, 4, 2)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 @ a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(sum(x @ a2, axis=-1), axis=-1), axis=-1)
    func2 = lambda x: sum(sum(sum(a1 @ x, axis=-1), axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


def test_div():
    # test tensor / scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1 / a2).sum()
    res.backward()

    func = lambda x: sum(sum(x / a2, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a1).to_np(), a1.grad)

    # test scalar / tensor
    a1_np, a2_np = np.random.rand(1), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np, requires_grad=False), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func = lambda x: sum(sum(a1 / x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a2).to_np(), a2.grad)

    # test tensor / tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x / a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 / x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1 / a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x / a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1 / x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


def test_pow():
    # test tensor ** scalar
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np, requires_grad=False)
    res = (a1**a2).sum()
    res.backward()

    func = lambda x: sum(sum(x**a2, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a1).to_np(), a1.grad)

    # test scalar ** tensor
    a1_np, a2_np = np.random.rand(1), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np, requires_grad=False), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func = lambda x: sum(sum(a1**x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, a2).to_np(), a2.grad)

    # test tensor ** tensor
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x**a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1**x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)

    # test with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = (a1**a2).sum()
    res.backward()

    func1 = lambda x: sum(sum(x**a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1**x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


def test_exp():
    input_np = np.random.rand(10)
    input = Tensor(input_np)
    res = sum(exp(input))
    res.backward()

    func = lambda x: sum(exp(x), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_log():
    input_np = np.random.rand(10)
    input = Tensor(input_np)
    res = sum(log(input))
    res.backward()

    func = lambda x: sum(log(x), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_gradient_from_multiple_paths():
    input_np = np.random.rand(10)
    input = Tensor(input_np)
    res = sum(input + input * input / exp(input))
    res.backward()

    func = lambda x: sum(x + x * x / exp(x), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_max():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(max(input, axis=-1))
    res.backward()

    func = lambda x: sum(max(x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)

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

    func = lambda x: sum(min(x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)

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
        softmax(input, axis=-1) * weight
    )  # must multiply a weight so that res != 3
    res.backward()

    func1 = lambda x: sum(sum(softmax(x, axis=-1) * weight, axis=-1), axis=-1)
    assert_close(numerical_grad(func1, input).to_np(), input.grad)
    func2 = lambda x: sum(sum(softmax(input, axis=-1) * x, axis=-1), axis=-1)
    assert_close(numerical_grad(func2, weight).to_np(), weight.grad)


def test_log_softmax():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(log_softmax(input, axis=-1))
    res.backward()

    func = lambda x: sum(sum(log_softmax(x, axis=-1), axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_reshape():
    input_np = np.random.rand(2, 6)
    input = Tensor(input_np)
    res = sum(softmax((input**2 + input).reshape(3, 4), axis=-1))
    res.backward()

    func = lambda x: sum(sum(softmax((x**2 + x).reshape(-1, 3, 4)), axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_squeeze():
    input_np = np.random.rand(2, 1, 3)
    input = Tensor(input_np)
    res = ((input.squeeze(axis=1) ** 2) + input.squeeze(axis=1)).sum()
    res.backward()

    func = lambda x: sum(
        sum(((x.squeeze(axis=2) ** 2) + x.squeeze(axis=2)), axis=-1), axis=-1
    )
    assert_close(numerical_grad(func, input).to_np(), input.grad)


def test_expand_dims():
    input_np = np.random.rand(2, 3)
    input = Tensor(input_np)
    res = ((input.expand_dims(axis=1) ** 2) + input.expand_dims(axis=1)).sum()
    res.backward()

    func = lambda x: sum(
        sum(
            sum(
                ((np.expand_dims(x, axis=2) ** 2) + np.expand_dims(x, axis=2)), axis=-1
            ),
            axis=-1,
        ),
        axis=-1,
    )
    assert_close(numerical_grad(func, input).to_np(), input.grad)
