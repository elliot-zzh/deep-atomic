import pytest
import numpy as np

from deep import *


def try_ufunc_call(ufunc, *args_np):
    args = [Tensor(i) if isinstance(i, np.ndarray) else i for i in args_np]
    res = ufunc(*args)
    assert (res.to_np() == ufunc(*args_np)).all()


def test_add():
    # general add
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.add, a1_np, a2_np)

    # add with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.add, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.add, a1_np, a2_np)

    # add with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.add, a1_np, a2_np)


def test_sub():
    # general sub
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.subtract, a1_np, a2_np)

    # sub with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.subtract, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.subtract, a1_np, a2_np)

    # sub with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.subtract, a1_np, a2_np)


def test_mul():
    # general mul
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.multiply, a1_np, a2_np)

    # mul with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.multiply, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.multiply, a1_np, a2_np)

    # mul with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.multiply, a1_np, a2_np)


def test_matmul():
    # general matmul
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4, 5)
    try_ufunc_call(np.matmul, a1_np, a2_np)

    # matmul with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.matmul, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4, 1)
    try_ufunc_call(np.matmul, a1_np, a2_np)


def test_div():
    # general div
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.divide, a1_np, a2_np)

    # div with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.divide, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.divide, a1_np, a2_np)

    # div with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.divide, a1_np, a2_np)


def test_pow():
    # general pow
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.power, a1_np, a2_np)

    # pow with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.power, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.power, a1_np, a2_np)

    # pow with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.power, a1_np, a2_np)


def test_exp():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = exp(input)
    assert (res.to_np() == np.exp(input_np)).all()


def test_log():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = log(input)
    assert (res.to_np() == np.log(input_np)).all()


def test_abs():
    input_np = np.random.rand(3, 4) - 0.5
    input = Tensor(input_np)
    res = abs(input)
    assert (res.to_np() == np.abs(input_np)).all()


def test_reshape():
    input_np = np.random.rand(2, 3, 4)
    input = Tensor(input_np)
    res = input.reshape(6, 4)
    assert (res.to_np() == np.reshape(input_np, (6, 4))).all()


def test_squeeze():
    input_np = np.random.rand(1, 3, 1, 4, 1)
    input = Tensor(input_np)
    res = input.squeeze(axis=(0, 2, 4))
    assert (res.to_np() == np.squeeze(input_np, axis=(0, 2, 4))).all()


def test_expand_dims():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = input.expand_dims(axis=0)
    assert (res.to_np() == np.expand_dims(input_np, axis=0)).all()


def test_sum():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)

    # test full reduction
    res = sum(input)
    assert res.to_np() == np.sum(input_np)

    # test selective reduction
    res = sum(input, axis=1)
    assert (res.to_np() == np.sum(input_np, axis=1)).all()


def test_max():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)

    # test full reduction
    # currently full reduction backcwarding not supported
    """
    res = max(input)
    assert res.to_np() == np.max(input_np)
    """

    # test selective reduction
    res = max(input, axis=1)
    assert (res.to_np() == np.max(input_np, axis=1)).all()


def test_min():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)

    # test full reduction
    # currently full reduction backcwarding not supported
    """
    res = min(input)
    assert res.to_np() == np.min(input_np)
    """

    # test selective reduction
    res = min(input, axis=1)
    assert (res.to_np() == np.min(input_np, axis=1)).all()


def test_argmax():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = argmax(input, axis=1)
    assert (res.to_np() == np.argmax(input_np, axis=1)).all()


def test_argmin():
    input_np = np.random.rand(3, 4)
    input = Tensor(input_np)
    res = argmin(input, axis=1)
    assert (res.to_np() == np.argmin(input_np, axis=1)).all()
