import logging

import numpy as np
import pytest

from deep_atomic import *

from .utils import *


def _test_unary_grad(init_fixture, op, *args, **kwargs):
    t = init_fixture
    res = op(t, *args, **kwargs).sum()
    res.backward()

    func = lambda x: op(x, *args, **kwargs).sum()

    assert_close(numerical_grad(func, t), t.grad)


def _test_binary_grad(init_fixture, op, *args, **kwargs):
    t1, t2 = init_fixture
    res = op(t1, t2, *args, **kwargs).sum()
    res.backward()

    func1 = lambda x: op(x, t2, *args, **kwargs).sum()
    func2 = lambda x: op(t1, x, *args, **kwargs).sum()

    assert_close(numerical_grad(func1, t1), t1.grad)
    assert_close(numerical_grad(func2, t2), t2.grad)


class TestArithmeticOps:
    @pytest.mark.parametrize("size1", [(3, 4), (2, 3, 4)])
    @pytest.mark.parametrize("size2", [(4, 5)])
    class TestMatMul:
        def test_grad(self, make_binary, size1, size2):
            _test_binary_grad(make_binary(size1=size1, size2=size2), lambda a, b: a @ b)

    class TestPow:
        def test_grad(self, make_binary):
            _test_binary_grad(
                make_binary(low1=0.0, high1=3.0, low2=-3.0, high2=3.0),
                lambda a, b: a**b,
            )

    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: a / b,
        ],
    )
    @pytest.mark.parametrize("size1", [(3, 4)])
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1)])
    class TestOthers:
        def test_grad(self, make_binary, op, size1, size2):
            _test_binary_grad(make_binary(size1=size1, size2=size2), op)


class TestUnaryMath:
    @pytest.mark.parametrize(
        "op", [exp, sin, cos, tan, arctan, sinh, cosh, tanh, arcsinh]
    )
    class TestDomainReal:
        def test_grad(self, make_unary, op):
            _test_unary_grad(make_unary(low=-5, high=5), op)

    @pytest.mark.parametrize("op", [arcsin, arccos, arctanh])
    class TestDomainMinus1ToPlus1:
        def test_grad(self, make_unary, op):
            _test_unary_grad(make_unary(low=-1, high=1), op)

    class TestLog:
        def test_grad(self, make_unary):
            _test_unary_grad(make_unary(low=0, high=1e3), lambda x: log(x))

    class TestArccosh:
        def test_grad(self, make_unary):
            _test_unary_grad(make_unary(low=1, high=1e3), arccosh)


class TestActivations:
    @pytest.mark.parametrize("op", [softmax, log_softmax])
    @pytest.mark.parametrize("axis", [0, 1])
    class TestSoftmax:
        def test_grad(self, unary, op, axis):
            _test_unary_grad(unary, op, axis=axis)

    @pytest.mark.parametrize("op", [sigmoid, relu, silu, gelu])
    class TestOthers:
        def test_grad(self, unary, op):
            _test_unary_grad(unary, op)


class TestRecductions:
    @pytest.mark.parametrize("op", [sum, max, min])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    class TestWithGrad:
        def test_grad(self, unary, op, axis, keepdims):
            _test_unary_grad(unary, op, axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize("op", [argmin, argmax])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    class TestWithoutGrad: ...


class TestComparison:
    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a == b,
            lambda a, b: a < b,
            lambda a, b: a <= b,
            lambda a, b: a > b,
            lambda a, b: a >= b,
            lambda a, b: a != b,
        ],
    )
    class TestResBool: ...

    @pytest.mark.parametrize("op", [fmax, fmin])
    class TestFMaxMin:
        def test_grad(self, binary, op):
            _test_binary_grad(binary, op)


class TestLogical: ...


class TestShapeOps:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    class TestExpandDims:
        def test_grad(self, unary, axis):
            _test_unary_grad(unary, expand_dims, axis=axis)

    @pytest.mark.parametrize("size", [(3, 4, 1), (3, 1, 4)])
    class TestSqueeze:
        def test_grad(self, make_unary, size):
            t = make_unary(size=size)
            axis = 0
            for i, v in enumerate(t.shape):
                if v == 1:
                    axis = i
            _test_unary_grad(t, squeeze, axis=axis)

    @pytest.mark.parametrize(
        "repeats_axis",
        [
            (2, None),
            (2, 0),
            ([1, 2, 3] * 4, None),
            ([1, 2, 3], 0),
        ],
    )
    class TestRepeat:
        def test_grad(self, unary, repeats_axis):
            repeats, axis = repeats_axis
            log_softmax_axis = -1 if axis is None else axis
            _test_unary_grad(
                unary,
                lambda x: log_softmax(
                    x.repeat(repeats, axis=axis), axis=log_softmax_axis
                ),
            )

    @pytest.mark.parametrize("reps", [(3,), (2, 3), (2, 2, 3)])
    class TestTile:
        def test_grad(self, binary, reps):
            # use more complex test to deepen the graph
            _test_binary_grad(binary, lambda a, b: log_softmax(a.tile(*reps)))

    @pytest.mark.parametrize("size1", [(3, 4), (2, 3, 4)])
    class TestWhere:
        def test_grad(self, rng, make_binary, size1):
            t1, _ = make_binary(size1=size1)
            condition = rng.choice([True, False], size=t1.shape)
            _test_binary_grad(
                make_binary(size1=size1),
                lambda a, b: where(condition, a, b),
            )

    @pytest.mark.parametrize("new_shape", [(2, 6), (3, 2, 2)])
    class TestReshape:
        def test_grad(self, unary, new_shape):
            # use more complex test to deepen the graph
            _test_unary_grad(
                unary, lambda x: log_softmax(x.reshape(new_shape), axis=-1)
            )


class TestSelection:
    class TestTopk:
        @pytest.mark.parametrize("axis", [0, 1])
        @pytest.mark.parametrize("kth", [1, 2])
        @pytest.mark.parametrize("largest", [True, False])
        def test_grad(self, unary, axis, kth, largest):
            _test_unary_grad(
                unary, lambda x: topk(x, axis=axis, kth=kth, largest=largest)[0]
            )
