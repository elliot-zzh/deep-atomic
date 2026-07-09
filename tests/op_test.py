import pytest
import numpy as np

from deep_atomic import *


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


def test_equal():
    # general eq
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.equal, a1_np, a2_np)

    # eq with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.equal, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.equal, a1_np, a2_np)

    # eq with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.equal, a1_np, a2_np)


def test_not_equal():
    # general not equal
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.not_equal, a1_np, a2_np)

    # not equal with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.not_equal, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.not_equal, a1_np, a2_np)

    # not equal with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.not_equal, a1_np, a2_np)


def test_greater():
    # general greater
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.greater, a1_np, a2_np)

    # greater with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.greater, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.greater, a1_np, a2_np)

    # greater with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.greater, a1_np, a2_np)


def test_greater_equal():
    # general greater equal
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.greater_equal, a1_np, a2_np)

    # greater equal with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.greater_equal, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.greater_equal, a1_np, a2_np)

    # greater equal with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.greater_equal, a1_np, a2_np)


def test_less():
    # general less
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.less, a1_np, a2_np)

    # less with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.less, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.less, a1_np, a2_np)

    # less with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.less, a1_np, a2_np)


def test_less_equal():
    # general less equal
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    try_ufunc_call(np.less_equal, a1_np, a2_np)

    # less equal with broadcast
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    try_ufunc_call(np.less_equal, a1_np, a2_np)
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 1)
    try_ufunc_call(np.less_equal, a1_np, a2_np)

    # less equal with scalar
    a1_np, a2_np = np.random.rand(3, 4), 5
    try_ufunc_call(np.less_equal, a1_np, a2_np)


def test_logical_not():
    # general logical not
    input_np = np.array([[True, False, True], [False, True, False]])
    input = Tensor(input_np)
    res = np.logical_not(input)
    assert (res.to_np() == np.logical_not(input_np)).all()


def test_logical_and():
    # general logical and
    a1_np = np.array([[True, False, True], [False, True, False]])
    a2_np = np.array([[True, False, True], [False, True, False]])
    try_ufunc_call(np.logical_and, a1_np, a2_np)

    # logical and with broadcast
    a2_np = np.array([[True], [False]])
    try_ufunc_call(np.logical_and, a1_np, a2_np)

    # logical and with scalar
    a2_np = True
    try_ufunc_call(np.logical_and, a1_np, a2_np)


def test_logical_or():
    # general logical or
    a1_np = np.array([[True, False, True], [False, True, False]])
    a2_np = np.array([[True, False, True], [False, True, False]])
    try_ufunc_call(np.logical_or, a1_np, a2_np)

    # logical or with broadcast
    a2_np = np.array([[True], [False]])
    try_ufunc_call(np.logical_or, a1_np, a2_np)

    # logical or with scalar
    a2_np = True
    try_ufunc_call(np.logical_or, a1_np, a2_np)


def test_logical_xor():
    # general logical xor
    a1_np = np.array([[True, False, True], [False, True, False]])
    a2_np = np.array([[True, False, True], [False, True, False]])
    try_ufunc_call(np.logical_xor, a1_np, a2_np)

    # logical xor with broadcast
    a2_np = np.array([[True], [False]])
    try_ufunc_call(np.logical_xor, a1_np, a2_np)

    # logical xor with scalar
    a2_np = True
    try_ufunc_call(np.logical_xor, a1_np, a2_np)
    
    
def test_all():
    # full reduction
    input_np = np.array([[True, False, True], [False, True, False]])
    input = Tensor(input_np)
    res = all(input)
    assert res.to_np() == np.all(input_np)
    
    # full along an axis
    res = all(input, axis=-1)
    assert (res.to_np() == np.all(input_np, axis=-1)).all()

def test_topk():
    input = Tensor(np.array([[1, 2, 3], [6, 4, 5]]))
    
    # largest
    expected_values = Tensor(np.array([[2, 3], [5, 6]]))
    expected_indices = Tensor(np.array([[1, 2], [2, 0]]))
    values, indices = topk(input, 2, axis=-1)
    assert (np.sort(values.to_np(), axis=-1) == expected_values).all() # sort to ensure order
    assert (np.take_along_axis(indices, np.argsort(values.to_np(), axis=-1), axis=-1) == expected_indices).all() # sort then gather to ensure correct mapping
    
    # smallest
    expected_values = Tensor(np.array([[1, 2], [4, 5]]))
    expected_indices = Tensor(np.array([[0, 1], [1, 2]]))
    values, indices = topk(input, 2, axis=-1, largest=False)
    assert (np.sort(values.to_np(), axis=-1) == expected_values).all()
    assert (np.take_along_axis(indices, np.argsort(values.to_np(), axis=-1), axis=-1) == expected_indices).all()
