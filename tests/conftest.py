import numpy as np
import pytest

import deep_atomic as da


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# tensor generation for unary ops


@pytest.fixture
def make_unary(rng):
    def _make(low=-1e3, high=1e3, size=(3, 4)):
        return da.Tensor(rng.uniform(low=low, high=high, size=size))

    return _make


@pytest.fixture
def unary(make_unary):
    return make_unary()


# tensor generation for binary ops


@pytest.fixture
def make_binary(rng):
    def _make(low1=-1e3, high1=1e3, low2=-1e3, high2=1e3, size1=(3, 4), size2=(3, 4)):
        return (
            da.Tensor(rng.uniform(low=low1, high=high1, size=size1)),
            da.Tensor(rng.uniform(low=low2, high=high2, size=size2)),
        )

    return _make


@pytest.fixture
def binary(make_binary):
    return make_binary()
