import logging
from tempfile import tempdir

import numpy as np
import pytest

import deep_atomic as da

from .utils import *


def _test_unary_forward(init_fixture, op, op_expected, *args, **kwargs):
    t = init_fixture
    actual = op(t, *args, **kwargs)
    expected = op_expected(t.to_np(), *args, **kwargs)
    assert actual.shape == expected.shape
    assert (actual == expected).all()


def _test_binary_forward(init_fixture, op, op_expected, *args, **kwargs):
    t1, t2 = init_fixture
    actual = op(t1, t2, *args, **kwargs)
    expected = op_expected(t1.to_np(), t2.to_np(), *args, **kwargs)
    assert actual.shape == expected.shape
    assert (actual == expected).all()


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
        def test_forward(self, make_binary, size1, size2):
            _test_binary_forward(
                make_binary(size1=size1, size2=size2),
                lambda a, b: a @ b,
                lambda a, b: a @ b,
            )

        def test_grad(self, make_binary, size1, size2):
            _test_binary_grad(make_binary(size1=size1, size2=size2), lambda a, b: a @ b)

    @pytest.mark.parametrize("size1", [(3, 4)])
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1), (4,)])
    class TestPow:
        def test_forward(self, make_binary, size1, size2):
            _test_binary_forward(
                make_binary(
                    low1=0.0, high1=3.0, low2=-3.0, high2=3.0, size1=size1, size2=size2
                ),
                lambda a, b: a**b,
                lambda a, b: a**b,
            )

        def test_grad(self, make_binary, size1, size2):
            _test_binary_grad(
                make_binary(
                    low1=0.0, high1=3.0, low2=-3.0, high2=3.0, size1=size1, size2=size2
                ),
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
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1), (4,)])
    class TestOthers:
        def test_forward(self, make_binary, op, size1, size2):
            _test_binary_forward(make_binary(size1=size1, size2=size2), op, op)

        def test_grad(self, make_binary, op, size1, size2):
            _test_binary_grad(make_binary(size1=size1, size2=size2), op)


class TestUnaryMath:
    @pytest.mark.parametrize(
        "op",
        [
            (da.exp, np.exp),
            (da.sin, np.sin),
            (da.sin, np.sin),
            (da.tan, np.tan),
            (da.arctan, np.arctan),
            (da.sinh, np.sinh),
            (da.cosh, np.cosh),
            (da.tanh, np.tanh),
            (da.arcsinh, np.arcsinh),
        ],
    )
    class TestDomainReal:
        def test_forward(self, make_unary, op):
            _test_unary_forward(make_unary(low=-5, high=5), *op)

        def test_grad(self, make_unary, op):
            _test_unary_grad(make_unary(low=-5, high=5), op[0])

    @pytest.mark.parametrize(
        "op", [(da.arcsin, np.arcsin), (da.arccos, np.arccos), (da.arctanh, np.arctanh)]
    )
    class TestDomainMinus1ToPlus1:
        def test_forward(self, make_unary, op):
            _test_unary_forward(make_unary(low=-1, high=1), *op)

        def test_grad(self, make_unary, op):
            _test_unary_grad(make_unary(low=-1, high=1), op[0])

    class TestLog:
        def test_forward(self, make_unary):
            _test_unary_forward(make_unary(low=0, high=1e3), da.log, np.log)

        def test_grad(self, make_unary):
            _test_unary_grad(make_unary(low=0, high=1e3), da.log)

    class TestArccosh:
        def test_forward(self, make_unary):
            _test_unary_forward(make_unary(low=1, high=1e3), da.arccosh, np.arccosh)

        def test_grad(self, make_unary):
            _test_unary_grad(make_unary(low=1, high=1e3), da.arccosh)


# these are not basic operations so we do not test their forward passing
# test their backward passing though, to test the correctness when deeper computational graph is concerned
class TestActivations:
    @pytest.mark.parametrize("op", [da.softmax, da.log_softmax])
    @pytest.mark.parametrize("axis", [0, 1])
    class TestSoftmax:
        def test_grad(self, unary, op, axis):
            _test_unary_grad(unary, op, axis=axis)

    @pytest.mark.parametrize("op", [da.sigmoid, da.relu, da.silu, da.gelu])
    class TestOthers:
        def test_grad(self, unary, op):
            _test_unary_grad(unary, op)


class TestRecductions:
    @pytest.mark.parametrize(
        "op", [(da.sum, np.sum), (da.max, np.max), (da.min, np.min)]
    )
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    class TestWithGrad:
        def test_forward(self, unary, op, axis, keepdims):
            _test_unary_forward(unary, *op, axis=axis, keepdims=keepdims)

        def test_grad(self, unary, op, axis, keepdims):
            _test_unary_grad(unary, op[0], axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize("op", [(da.argmin, np.argmin), (da.argmax, np.argmax)])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    class TestWithoutGrad:
        def test_forward(self, unary, op, axis, keepdims):
            _test_unary_forward(unary, *op, axis=axis, keepdims=keepdims)


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
    @pytest.mark.parametrize("size1", [(3, 4)])
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1), (4,)])
    class TestResBool:
        def test_forward(self, make_binary, op, size1, size2):
            _test_binary_forward(make_binary(size1=size1, size2=size2), op, op)

    @pytest.mark.parametrize("op", [(da.fmax, np.fmax), (da.fmin, np.fmin)])
    @pytest.mark.parametrize("size1", [(3, 4)])
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1), (4,)])
    class TestFMaxMin:
        def test_forward(self, make_binary, op, size1, size2):
            _test_binary_forward(make_binary(size1=size1, size2=size2), *op)

        def test_grad(self, make_binary, op, size1, size2):
            _test_binary_grad(make_binary(size1=size1, size2=size2), op[0])


class TestLogical:
    @pytest.mark.parametrize(
        "op",
        [
            (da.logical_and, np.logical_and),
            (da.logical_or, np.logical_or),
            (da.logical_xor, np.logical_xor),
            lambda a, b: a | b,
            lambda a, b: a & b,
            lambda a, b: a ^ b,
        ],
    )
    @pytest.mark.parametrize("size1", [(3, 4)])
    @pytest.mark.parametrize("size2", [(3, 4), (3, 1), (4,)])
    class TestBinary:
        def test_forward(self, rng, op, size1, size2):
            binary = (
                da.Tensor(rng.choice([True, False], size=size1)),
                da.Tensor(rng.choice([True, False], size=size2)),
            )
            if isinstance(op, tuple):
                _test_binary_forward(binary, *op)
            else:
                _test_binary_forward(binary, op, op)

    class TestLogicalNot:
        def test_forward(self, rng):
            unary = da.Tensor(rng.choice([True, False], size=(3, 4)))
            _test_unary_forward(unary, da.logical_not, np.logical_not)

    class TestBitwiseNot:
        def test_forward(self, rng):
            unary = da.Tensor(rng.choice([True, False], size=(3, 4)))
            op = lambda x: ~x
            _test_unary_forward(unary, op, op)

    @pytest.mark.parametrize("op", [(da.any, np.any), (da.all, np.all)])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    class TestReductions:
        def test_forward(self, rng, op, axis, keepdims):
            unary = da.Tensor(rng.choice([True, False], size=(3, 4)))
            _test_unary_forward(unary, *op, axis=axis, keepdims=keepdims)


class TestShapeOps:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    class TestExpandDims:
        def test_forward(self, unary, axis):
            _test_unary_forward(unary, da.expand_dims, np.expand_dims, axis=axis)

        def test_grad(self, unary, axis):
            _test_unary_grad(unary, da.expand_dims, axis=axis)

    @pytest.mark.parametrize("size", [(3, 4, 1), (3, 1, 4)])
    class TestSqueeze:
        def test_forward(self, make_unary, size):
            t = make_unary(size=size)
            axis = 0
            for i, v in enumerate(t.shape):
                if v == 1:
                    axis = i
            _test_unary_forward(t, da.squeeze, np.squeeze, axis=axis)

        def test_grad(self, make_unary, size):
            t = make_unary(size=size)
            axis = 0
            for i, v in enumerate(t.shape):
                if v == 1:
                    axis = i
            _test_unary_grad(t, da.squeeze, axis=axis)

    @pytest.mark.parametrize(
        "repeats, axis",
        [
            (2, None),
            (2, 0),
            ([1, 2, 3] * 4, None),
            ([1, 2, 3], 0),
        ],
    )
    class TestRepeat:
        def test_forward(self, unary, repeats, axis):
            _test_unary_forward(unary, da.repeat, np.repeat, repeats=repeats, axis=axis)

        def test_grad(self, unary, repeats, axis):
            log_softmax_axis = -1 if axis is None else axis
            _test_unary_grad(
                unary,
                lambda x: da.log_softmax(
                    x.repeat(repeats, axis=axis), axis=log_softmax_axis
                ),
            )

    @pytest.mark.parametrize("reps", [(3,), (2, 3), (2, 2, 3)])
    class TestTile:
        def test_forward(self, unary, reps):
            _test_unary_forward(unary, da.tile, np.tile, reps=reps)

        def test_grad(self, unary, reps):
            # use more complex test to deepen the graph
            _test_unary_grad(unary, lambda a: da.log_softmax(a.tile(*reps)))

    @pytest.mark.parametrize("size1", [(3, 4), (2, 3, 4)])
    class TestWhere:
        def test_forward(self, rng, make_binary, size1):
            t1, _ = make_binary(size1=size1)
            condition = da.Tensor(rng.choice([True, False], size=t1.shape))
            _test_binary_forward(
                make_binary(size1=size1),
                lambda a, b: da.where(condition, a, b),
                lambda a, b: np.where(condition.to_np(), a, b),
            )

        def test_grad(self, rng, make_binary, size1):
            t1, _ = make_binary(size1=size1)
            condition = da.Tensor(rng.choice([True, False], size=t1.shape))
            _test_binary_grad(
                make_binary(size1=size1),
                lambda a, b: da.where(condition, a, b),
            )

    @pytest.mark.parametrize("target_shape", [(2, 6), (3, 2, 2)])
    class TestReshape:
        def test_forward(self, unary, target_shape):
            _test_unary_forward(
                unary,
                lambda x: da.reshape(x, target_shape=target_shape),
                lambda x: np.reshape(x, shape=target_shape),
            )

        def test_grad(self, unary, target_shape):
            # use more complex test to deepen the graph
            _test_unary_grad(
                unary, lambda x: da.log_softmax(x.reshape(target_shape), axis=-1)
            )


class TestSelection:
    class TestTopk:
        class test_forward:
            # numpy does not have direct equivalence to topk
            # and given topk's complexity, we hardwired this test
            def test_largest(self):
                t = da.Tensor(np.array([[1, 2, 3], [6, 4, 5]]))

                # largest
                expected_values = da.Tensor(np.array([[2, 3], [5, 6]]))
                expected_indices = da.Tensor(np.array([[1, 2], [2, 0]]))
                values, indices = da.topk(t, 2, axis=-1)
                assert (
                    np.sort(values.to_np(), axis=-1) == expected_values
                ).all()  # sort to ensure order
                assert (
                    np.take_along_axis(
                        indices, np.argsort(values.to_np(), axis=-1), axis=-1
                    )
                    == expected_indices
                ).all()  # sort then gather to ensure correct mapping

            def test_smallest(self):
                t = da.Tensor(np.array([[1, 2, 3], [6, 4, 5]]))

                # smallest
                expected_values = da.Tensor(np.array([[1, 2], [4, 5]]))
                expected_indices = da.Tensor(np.array([[0, 1], [1, 2]]))
                values, indices = da.topk(t, 2, axis=-1, largest=False)
                assert (np.sort(values.to_np(), axis=-1) == expected_values).all()
                assert (
                    np.take_along_axis(
                        indices, np.argsort(values.to_np(), axis=-1), axis=-1
                    )
                    == expected_indices
                ).all()

        @pytest.mark.parametrize("axis", [0, 1])
        @pytest.mark.parametrize("kth", [1, 2])
        @pytest.mark.parametrize("largest", [True, False])
        def test_grad(self, unary, axis, kth, largest):
            _test_unary_grad(
                unary, lambda x: da.topk(x, axis=axis, kth=kth, largest=largest)[0]
            )
