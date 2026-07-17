import numpy as np
import pytest

import deep_atomic as da
from deep_atomic.nn import Linear, MSELoss
from deep_atomic.optim import SGD, Adam, AdamW, Muon

from .utils import assert_close

N_SAMPLES = 32
IN_D = 16
OUT_D = 16


@pytest.fixture
def optim_task(rng):
    """Generate a random Linear regression task with known optimum."""
    inp = da.Tensor(rng.standard_normal((N_SAMPLES, IN_D)) * 0.5, requires_grad=False)
    w_opt = da.Tensor(rng.standard_normal((IN_D, OUT_D)) * 0.3, requires_grad=False)
    b_opt = da.Tensor(rng.standard_normal(OUT_D) * 0.2, requires_grad=False)
    return {
        "input": inp,
        "w_opt": w_opt,
        "b_opt": b_opt,
        "target_no_bias": inp @ w_opt,
        "target_with_bias": inp @ w_opt + b_opt,
    }


class TestOptimizers:
    """Verify all optimizers converge on Linear regression with known optimum."""

    def _run(self, optim_cls, optim_kwargs, steps, task, *, bias):
        """Common harness: build model, train, return (final_loss, weight, bias)."""
        np.random.seed(42)
        target = task["target_with_bias"] if bias else task["target_no_bias"]
        model = Linear(IN_D, OUT_D, bias=bias)
        loss_fn = MSELoss()
        opt = optim_cls(model.parameters(), **optim_kwargs)

        final_loss = None
        for _ in range(steps):
            output = model(task["input"])
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            opt.zero_grad()
            final_loss = loss.to_np().item()

        return final_loss, model.weight, model.bias

    @pytest.mark.parametrize(
        "momentum, nesterov",
        [(0, False), (0.9, False), (0.9, True)],
        ids=["sgd", "sgdm", "sgd_nesterov"],
    )
    def test_sgd(self, optim_task, momentum, nesterov):
        final_loss, weight, bias = self._run(
            SGD,
            dict(lr=0.5, momentum=momentum, nesterov=nesterov),
            steps=5000,
            task=optim_task,
            bias=True,
        )
        assert -1e-4 < final_loss < 1e-4, f"loss={final_loss:.2e}"
        assert_close(optim_task["w_opt"].to_np(), weight.to_np(), atol=1e-2)
        assert_close(optim_task["b_opt"].to_np(), bias.to_np(), atol=1e-2)

    def test_adam(self, optim_task):
        final_loss, weight, bias = self._run(
            Adam,
            dict(lr=0.05),
            steps=4000,
            task=optim_task,
            bias=True,
        )
        assert -1e-4 < final_loss < 1e-4, f"loss={final_loss:.2e}"
        assert_close(optim_task["w_opt"].to_np(), weight.to_np(), atol=1e-2)
        assert_close(optim_task["b_opt"].to_np(), bias.to_np(), atol=1e-2)

    def test_adamw(self, optim_task):
        final_loss, weight, bias = self._run(
            AdamW,
            dict(lr=0.05, weight_decay=0),
            steps=4000,
            task=optim_task,
            bias=True,
        )
        assert -1e-4 < final_loss < 1e-4, f"loss={final_loss:.2e}"
        assert_close(optim_task["w_opt"].to_np(), weight.to_np(), atol=1e-2)
        assert_close(optim_task["b_opt"].to_np(), bias.to_np(), atol=1e-2)

    @pytest.mark.parametrize(
        "adjust_lr_fn",
        ["original", "match_rms_adamw"],
        ids=["muon_original", "muon_match_rms_adamw"],
    )
    def test_muon(self, optim_task, adjust_lr_fn):
        final_loss, weight, bias = self._run(
            Muon,
            dict(lr=0.01, weight_decay=0, adjust_lr_fn=adjust_lr_fn),
            steps=3000,
            task=optim_task,
            bias=False,
        )
        assert -1e-4 < final_loss < 1e-4, f"loss={final_loss:.2e}"
        assert_close(optim_task["w_opt"].to_np(), weight.to_np(), atol=1e-2)
