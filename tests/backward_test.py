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


"""
def test_matmul():
    return True
"""


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


"""
def test_pow():
    return True

def test_exp():
    return True

def test_log():
    return True

def test_composition():
    return True
    
"""
