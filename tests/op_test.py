import pytest
import numpy as np

from deep import *

# TODO: test more types and shapes (including broadcasting)


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
    base_np = np.random.rand(3, 4)
    exponent = 2.5
    base = Tensor(base_np)
    res = base**exponent
    assert (res.to_np() == base_np**exponent).all()


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
