import numpy as np
import pytest

import deep_atomic as da
from deep_atomic.nn import (
    Buffer,
    BufferList,
    Linear,
    Module,
    ModuleList,
    MSELoss,
    Parameter,
    ParameterList,
    ReLU,
    Sequential,
    Softmax,
)

from .utils import assert_close, numerical_grad


def _make_identity_module():
    """Factory for a minimal identity Module."""

    class M(Module):
        def forward(self, x):
            return x

    return M()


class TestBuffer:
    def test_requires_grad_always_false(self):
        """Buffer.requires_grad is False even when passed requires_grad=True."""
        buf = Buffer(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        assert buf.requires_grad is False

    def test_is_buffer_and_tensor(self):
        buf = Buffer(np.array([1, 2, 3]))
        assert isinstance(buf, Buffer)
        assert isinstance(buf, da.Tensor)


MEMBER_PARAMS = [
    (Parameter, "named_parameters", "parameters"),
    (Buffer, "named_buffers", "buffers"),
]  # for TestModuleNamedMembers


class TestModule:
    class TestModuleSetAttr:
        """__setattr__ routes Parameter/Buffer/Module into their tracking dicts."""

        @pytest.mark.parametrize(
            "cls, dict_name",
            [(Parameter, "_parameters"), (Buffer, "_buffers")],
        )
        def test_member_routing(self, cls, dict_name):
            m = _make_identity_module()
            v = cls(np.array([1.0]))
            m.w = v
            assert "w" in getattr(m, dict_name)
            assert getattr(m, dict_name)["w"] is v

        def test_module_routing(self):
            child = _make_identity_module()
            parent = _make_identity_module()
            parent.child = child
            assert "child" in parent._modules
            assert parent._modules["child"] is child

        def test_plain_value_not_tracked(self):
            m = _make_identity_module()
            m.x = 3
            m.label = "hello"
            assert "x" not in m._parameters
            assert "x" not in m._buffers
            assert "x" not in m._modules
            assert "label" not in m._parameters
            assert "label" not in m._buffers
            assert "label" not in m._modules

    class TestModuleStateDict:
        def test_flat(self):
            """Flat module: state_dict contains every Parameter + Buffer keyed by name."""

            class M(Module):
                def __init__(self):
                    super().__init__()
                    self.w = Parameter(np.array([1.0, 2.0]))
                    self.buf = Buffer(np.array([3.0]))

                def forward(self, x):
                    return x

            m = M()
            sd = m.state_dict()
            assert set(sd.keys()) == {"w", "buf"}
            assert sd["w"] is m.w
            assert sd["buf"] is m.buf

        def test_nested(self):
            """Nested: state_dict flattens recursively with dotted prefix."""

            class Leaf(Module):
                def __init__(self):
                    super().__init__()
                    self.w = Parameter(np.array([1.0]))

                def forward(self, x):
                    return x

            class Mid(Module):
                def __init__(self):
                    super().__init__()
                    self.b = Leaf()

                def forward(self, x):
                    return x

            class Root(Module):
                def __init__(self):
                    super().__init__()
                    self.a = Mid()

                def forward(self, x):
                    return x

            r = Root()
            sd = r.state_dict()
            assert sd == {"a.b.w": r.a.b.w}

    class TestModuleNamedModules:
        def test_no_submodules(self):
            """Module with no submodules yields only self."""

            m = _make_identity_module()
            result = list(m.named_modules())
            assert len(result) == 1
            assert result[0][0] == ""
            assert result[0][1] is m

        def test_recurse(self):
            """Yields self + all descendants with dotted prefixes."""

            inner = _make_identity_module()

            class Outer(Module):
                def __init__(self):
                    super().__init__()
                    self.a = inner

                def forward(self, x):
                    return x

            outer = Outer()
            result = list(outer.named_modules())
            assert len(result) == 2
            assert result[0][0] == ""
            assert result[0][1] is outer
            assert result[1][0] == "a"
            assert result[1][1] is outer.a

        def test_remove_duplicate(self):
            """remove_duplicate=True → shared submodule yielded once."""

            shared = _make_identity_module()

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.left = shared
                    self.right = shared

                def forward(self, x):
                    return x

            p = Parent()
            result = list(p.named_modules(remove_duplicate=True))
            prefixes = [prefix for prefix, _ in result]
            assert prefixes.count("") == 1  # self
            assert prefixes.count("left") == 1

        def test_modules_matches_named_modules(self):
            """modules() values match named_modules() values (names stripped)."""

            inner = _make_identity_module()

            class Outer(Module):
                def __init__(self):
                    super().__init__()
                    self.a = inner

                def forward(self, x):
                    return x

            outer = Outer()
            named_mods = [m for _, m in outer.named_modules()]
            mods = list(outer.modules())
            assert mods == named_mods

    @pytest.mark.parametrize("cls, named, bare", MEMBER_PARAMS)
    class TestModuleNamedMembers:
        def test_flat(self, cls, named, bare):
            class M(Module):
                def __init__(self):
                    super().__init__()
                    self.a = cls(np.array([1.0]))
                    self.b = cls(np.array([2.0]))

                def forward(self, x):
                    return x

            m = M()
            result = list(getattr(m, named)())
            assert len(result) == 2
            names = {name for name, _ in result}
            assert names == {"a", "b"}

        def test_nested_recurse(self, cls, named, bare):
            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.w = cls(np.array([1.0]))

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()
                    self.b = cls(np.array([2.0]))

                def forward(self, x):
                    return x

            p = Parent()
            result = list(getattr(p, named)())
            names = {name for name, _ in result}
            assert names == {"b", "child.w"}

        def test_no_recurse(self, cls, named, bare):
            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.w = cls(np.array([1.0]))

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()
                    self.b = cls(np.array([2.0]))

                def forward(self, x):
                    return x

            p = Parent()
            result = list(getattr(p, named)(recurse=False))
            names = {name for name, _ in result}
            assert names == {"b"}

        def test_remove_duplicate(self, cls, named, bare):
            shared = cls(np.array([1.0]))

            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.m = shared

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.a = Child()
                    self.b = Child()

                def forward(self, x):
                    return x

            p = Parent()
            result = list(getattr(p, named)(remove_duplicate=True))
            names = [name for name, _ in result]
            assert names.count("a.m") == 1
            assert "b.m" not in names

        def test_bare_matches_named(self, cls, named, bare):
            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.w = cls(np.array([1.0]))

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()
                    self.b = cls(np.array([2.0]))

                def forward(self, x):
                    return x

            p = Parent()
            named_vals = [v for _, v in getattr(p, named)()]
            bare_vals = list(getattr(p, bare)())
            assert bare_vals == named_vals

        def test_empty(self, cls, named, bare):
            m = _make_identity_module()
            assert list(getattr(m, named)()) == []
            assert list(getattr(m, bare)()) == []

    class TestModuleTrainEval:
        def test_train(self):
            """train() sets requires_grad=True on all params including nested."""

            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.w = Parameter(np.array([1.0]))
                    self.w.requires_grad = False

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()
                    self.b = Parameter(np.array([2.0]))
                    self.b.requires_grad = False

                def forward(self, x):
                    return x

            p = Parent()
            p.train()
            params = list(p.parameters())
            assert all(param.requires_grad for param in params)

        def test_eval(self):
            """eval() sets requires_grad=False on all params including nested."""

            class Child(Module):
                def __init__(self):
                    super().__init__()
                    self.w = Parameter(np.array([1.0]))

                def forward(self, x):
                    return x

            class Parent(Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()
                    self.b = Parameter(np.array([2.0]))

                def forward(self, x):
                    return x

            p = Parent()
            p.eval()
            params = list(p.parameters())
            assert all(not param.requires_grad for param in params)

        def test_buffers_unaffected(self):
            """Buffers keep requires_grad=False regardless of train/eval."""

            class M(Module):
                def __init__(self):
                    super().__init__()
                    self.buf = Buffer(np.array([0.0]))

                def forward(self, x):
                    return x

            m = M()
            assert m.buf.requires_grad is False
            m.train()
            assert m.buf.requires_grad is False
            m.eval()
            assert m.buf.requires_grad is False

    class TestModuleCall:
        def test_call_returns_forward_result(self):
            """__call__ delegates to forward and returns its result."""

            class M(Module):
                def forward(self, x):
                    return x * 2

            m = M()
            result = m(da.Tensor(np.array([1.0, 2.0])))
            expected = np.array([2.0, 4.0])
            assert (result.to_np() == expected).all()


LIST_PARAMS = [
    (
        ParameterList,
        lambda: Parameter(np.array([1.0])),
        "_parameters",
        "named_parameters",
        True,
    ),
    (ModuleList, _make_identity_module, "_modules", "named_modules", False),
    (BufferList, lambda: Buffer(np.array([1.0])), "_buffers", "named_buffers", True),
]


@pytest.mark.parametrize("list_cls, make_member, _d, _n, auto_convert", LIST_PARAMS)
class TestListContainers:
    def test_len(self, list_cls, make_member, _d, _n, auto_convert):
        lst = list_cls([make_member(), make_member()])
        assert len(lst) == 2

    def test_getitem(self, list_cls, make_member, _d, _n, auto_convert):
        m0, m1 = make_member(), make_member()
        lst = list_cls([m0, m1])
        assert lst[0] is m0
        assert lst[1] is m1
        assert lst[-1] is m1
        assert lst[-2] is m0
        with pytest.raises(IndexError):
            _ = lst[2]
        with pytest.raises(IndexError):
            _ = lst[-3]

    def test_setitem(self, list_cls, make_member, _d, _n, auto_convert):
        m0, m1 = make_member(), make_member()
        lst = list_cls([m0])
        lst[0] = m1
        assert lst[0] is m1

    def test_setitem_auto_converts(self, list_cls, make_member, _d, _n, auto_convert):
        if not auto_convert:
            pytest.skip("auto-convert only applies to ParameterList and BufferList")
        t = da.Tensor(np.array([1.0]))
        lst = list_cls([make_member()])
        lst[0] = t
        assert isinstance(lst[0], (Parameter, Buffer))

    def test_iter(self, list_cls, make_member, _d, _n, auto_convert):
        members = [make_member(), make_member()]
        lst = list_cls(members)
        assert list(lst) == members

    def test_append(self, list_cls, make_member, _d, _n, auto_convert):
        m = make_member()
        lst = list_cls([make_member()])
        initial_len = len(lst)
        lst.append(m)
        assert len(lst) == initial_len + 1
        assert lst[-1] is m

    def test_extend(self, list_cls, make_member, _d, _n, auto_convert):
        m0, m1 = make_member(), make_member()
        lst = list_cls([make_member()])
        lst.extend([m0, m1])
        assert len(lst) == 3
        assert lst[1] is m0
        assert lst[2] is m1

    def test_parent_routing(self, list_cls, make_member, _d, _n, auto_convert):
        m = make_member()

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.items = list_cls([m])

            def forward(self, x):
                return x

        p = Parent()
        entries = dict(getattr(p, _n)())
        assert "items.0" in entries
        assert entries["items.0"] is m


class TestLinear:
    def test_weight_shape(self):
        lin = Linear(3, 5)
        assert lin.weight.shape == (3, 5)

    def test_bias_shape(self):
        lin = Linear(3, 5, bias=True)
        assert lin.bias.shape == (5,)

    def test_bias_is_none_when_disabled(self):
        lin = Linear(3, 5, bias=False)
        assert lin.bias is None

    def test_forward_with_bias(self):
        lin = Linear(3, 5, bias=True)
        x_np = np.random.randn(2, 3).astype(np.float64)
        x = da.Tensor(x_np)
        out = lin(x)
        expected = x_np @ lin.weight.to_np() + lin.bias.to_np()
        assert out.shape == (2, 5)
        assert (out.to_np() == expected).all()

    def test_forward_without_bias(self):
        lin = Linear(3, 5, bias=False)
        x_np = np.random.randn(2, 3).astype(np.float64)
        x = da.Tensor(x_np)
        out = lin(x)
        expected = x_np @ lin.weight.to_np()
        assert out.shape == (2, 5)
        assert (out.to_np() == expected).all()


# --- numpy equivalents for each activation ---


def _relu_np(z):
    return np.maximum(z, 0)


def _softmax_np(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize(
    "act_cls, act_np",
    [
        (ReLU, _relu_np),
        (Softmax, _softmax_np),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
class TestSequentialWithActivations:
    def test_forward(self, act_cls, act_np, bias):
        lin = Linear(3, 5, bias=bias)
        seq = Sequential(lin, act_cls())

        x_np = np.random.randn(2, 3).astype(np.float64)
        x = da.Tensor(x_np)

        out = seq(x)

        z = x_np @ lin.weight.to_np()
        if bias:
            z = z + lin.bias.to_np()
        expected = act_np(z)
        assert_close(expected, out.to_np())

    def test_grad(self, act_cls, act_np, bias):
        lin = Linear(3, 5, bias=bias)
        seq = Sequential(lin, act_cls())
        lossf = MSELoss()

        x_np = np.random.randn(2, 3).astype(np.float64)
        x = da.Tensor(x_np, requires_grad=True)
        y_np = np.random.randn(2, 5).astype(np.float64)
        y = da.Tensor(y_np, requires_grad=False)

        out = lossf(seq(x), y)
        out.backward()

        def func(inp):
            return da.mse_loss(seq(inp), y)

        assert_close(numerical_grad(func, x), x.grad)
