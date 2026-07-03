import pytest
from deep import *
import numpy as np

def test_init():
    assert (np.array([1,2,3]) == Tensor([1,2,3]).to_np()).all()
    assert (np.array([[1,2], [3,4]]) == Tensor([[1,2], [3,4]]).to_np()).all()
    assert (np.array([[1,2], [3,4]]) == Tensor([[1,2], [3,4]], requires_grad=True).to_np()).all()
    # test float
    assert (np.array([1.0,2.0,3.0]) == Tensor([1.0,2.0,3.0]).to_np()).all()
    assert (np.array([[1.0,2.0], [3.0,4.0]]) == Tensor([[1.0,2.0], [3.0,4.0]]).to_np()).all()
    # test conversion from ndarray
    assert (np.array([1,2,3]) == Tensor(np.array([1,2,3])).to_np()).all()
    assert (np.array([[1,2], [3,4]]) == Tensor(np.array([[1,2], [3,4]])).to_np()).all()
    assert (np.array([[1,2], [3,4]]) == Tensor(np.array([[1,2], [3,4]]), requires_grad=True).to_np()).all()
    assert (np.array([1.0,2.0,3.0]) == Tensor(np.array([1.0,2.0,3.0])).to_np()).all()
    assert (np.array([[1.0,2.0], [3.0,4.0]]) == Tensor(np.array([[1.0,2.0], [3.0,4.0]])).to_np()).all()
