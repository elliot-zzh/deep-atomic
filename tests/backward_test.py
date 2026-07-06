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


def test_exp():
    input_np = np.random.rand(10)
    input = Tensor(input_np)
    res = sum(exp(input))
    res.backward()

    func = lambda x: sum(exp(x), axis=-1)
    assert_close(numerical_grad(func, input), input.grad)


def test_log():
    input_np = np.random.rand(10)
    input = Tensor(input_np)
    res = sum(log(input))
    res.backward()

    func = lambda x: sum(log(x), axis=-1)
    assert_close(numerical_grad(func, input), input.grad)


def test_pow():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = sum(a1**a2)
    res.backward()

    func1 = lambda x: sum(sum(x**a2, axis=-1), axis=-1)
    func2 = lambda x: sum(sum(a1**x, axis=-1), axis=-1)

    assert_close(numerical_grad(func1, a1).to_np(), a1.grad)
    assert_close(numerical_grad(func2, a2).to_np(), a2.grad)


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


def test_min():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = sum(min(input, axis=-1))
    res.backward()

    func = lambda x: sum(min(x, axis=-1), axis=-1)
    assert_close(numerical_grad(func, input).to_np(), input.grad)


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
