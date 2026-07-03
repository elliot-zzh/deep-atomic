import pytest
import numpy as np

from deep import *


def test_add():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 + a2
    assert (res.to_np() == a1_np + a2_np).all()

def test_sub():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 - a2
    assert (res.to_np() == a1_np - a2_np).all()

def test_mul():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 * a2
    assert (res.to_np() == a1_np * a2_np).all()
    
def test_mul_broadcast():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 * a2
    assert (res.to_np() == a1_np * a2_np).all()

def test_matmul():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(4, 5)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 @ a2
    assert (res.to_np() == a1_np @ a2_np).all()


def test_div():
    a1_np, a2_np = np.random.rand(3, 4), np.random.rand(3, 4)
    a1, a2 = Tensor(a1_np), Tensor(a2_np)
    res = a1 / a2
    assert (res.to_np() == a1_np / a2_np).all()

def test_pow():
    base_np = np.random.rand(3, 4)
    exponent = 2.5
    base = Tensor(base_np)
    res = base ** exponent
    assert (res.to_np() == base_np ** exponent).all()


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

