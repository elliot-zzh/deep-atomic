from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from . import nn
from .tensor import *


class Optimizer(ABC):
    def __init__(self, params, defaults):
        params = list(params)
        first = params[0]
        if isinstance(first, nn.Parameter):
            self.param_groups = [{"params": params, **defaults}]
        elif isinstance(first, dict):
            self.param_groups = []
            for group in params:
                self.param_groups.append({**defaults, **group})
        else:
            raise ValueError(f"wrong params type. params: {params}")
        self.state = defaultdict(dict)

    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = np.zeros(param.shape)

    # TODO: support closure in the future?
    @abstractmethod
    def step(self):
        pass

    # TODO: to be implemented
    def state_dict(self): ...
    def load_state_dict(self): ...


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        nesterov=False,
        maximize=False,
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            params = group["params"]

            for param in params:
                # following pytorch implementation
                # https://docs.pytorch.org/docs/2.13/generated/torch.optim.sgd.SGD_class.html

                state = self.state[param]

                if maximize:
                    grad = -param.grad
                else:
                    grad = param.grad

                if weight_decay != 0:
                    grad += weight_decay * param.to_np()

                if momentum != 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad
                    else:
                        state["momentum_buffer"] *= momentum
                        state["momentum_buffer"] += (1 - dampening) * grad

                    if nesterov:
                        grad += momentum * state["momentum_buffer"]
                    else:
                        grad = state["momentum_buffer"]

                # disable graph construct
                # we do not set param.requires_grad since it will clear up the grad
                param._requires_grad = False
                np.subtract(param.to_np(), lr * grad, out=param.to_np())
                param._requires_grad = True


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta=(0.9, 0.999),
        eps=1e-8,
        maximize=False,
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            params = group["params"]

            for param in params:
                # following pytorch implementation
                # https://docs.pytorch.org/docs/2.13/generated/torch.optim.Adam.html

                state = self.state[param]

                if maximize:
                    grad = -param.grad
                else:
                    grad = param.grad

                if weight_decay != 0:
                    grad += weight_decay * param.to_np()

                if "t" not in state:
                    state["t"] = 1
                else:
                    state["t"] += 1
                t = state["t"]

                if "m_buffer" not in state:
                    state["m_buffer"] = np.zeros(param.shape)
                if "v_buffer" not in state:
                    state["v_buffer"] = np.zeros(param.shape)

                m_buffer = state["m_buffer"]
                v_buffer = state["v_buffer"]

                m_buffer *= beta1
                v_buffer *= beta2
                m_buffer += (1 - beta1) * grad
                v_buffer += (1 - beta2) * (grad**2)

                lr_ = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                grad = m_buffer / (v_buffer**0.5 + eps)

                # disable graph construct
                # we do not set param.requires_grad since it will clear up the grad
                param._requires_grad = False
                np.subtract(param.to_np(), lr_ * grad, out=param.to_np())
                param._requires_grad = True


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta=(0.9, 0.999),
        eps=1e-8,
        maximize=False,
        weight_decay=0.01,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            params = group["params"]

            for param in params:
                # following pytorch implementation
                # https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html

                state = self.state[param]

                if maximize:
                    grad = -param.grad
                else:
                    grad = param.grad

                if "t" not in state:
                    state["t"] = 1
                else:
                    state["t"] += 1
                t = state["t"]

                if "m_buffer" not in state:
                    state["m_buffer"] = np.zeros(param.shape)
                if "v_buffer" not in state:
                    state["v_buffer"] = np.zeros(param.shape)

                m_buffer = state["m_buffer"]
                v_buffer = state["v_buffer"]

                m_buffer *= beta1
                v_buffer *= beta2
                m_buffer += (1 - beta1) * grad
                v_buffer += (1 - beta2) * (grad**2)

                grad = m_buffer / (v_buffer**0.5 + eps)

                # disable graph construct
                # we do not set param.requires_grad since it will clear up the grad
                param._requires_grad = False
                np.multiply(param.to_np(), 1 - lr * weight_decay, out=param.to_np())
                lr_ = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                np.subtract(param.to_np(), lr_ * grad, out=param.to_np())
                param._requires_grad = True


class Muon(Optimizer):
    # follow pytorch's convention
    # https://docs.pytorch.org/docs/2.13/generated/torch.optim.Muon.html

    @staticmethod
    def _newton_schulz(G, coefficients, eps, steps):
        assert G.ndim == 2
        a, b, c = coefficients
        X = G
        X /= (X**2).sum() ** 0.5 + eps
        if G.shape[0] > G.shape[1]:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X

    def __init__(
        self,
        params,
        lr=0.001,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn="original",
    ):
        # we do not import op.py in this file, so `max` is a python built-in func here
        if adjust_lr_fn == "original":
            adjust_lr_fn = lambda lr, A, B: lr * max(1, A / B) ** 0.5
        elif adjust_lr_fn == "match_rms_adamw":
            adjust_lr_fn = lambda lr, A, B: 0.2 * lr * max(A, B) ** 0.5
        else:
            raise ValueError(
                f"do not support argument value adjust_lr_fn={adjust_lr_fn}"
            )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            params = group["params"]

            for param in params:
                grad = param.grad
                state = self.state[param]

                if "m_buffer" not in state:
                    state["m_buffer"] = np.zeros(param.shape)
                m_buffer = state["m_buffer"]

                m_buffer *= momentum
                m_buffer += grad

                if nesterov:
                    grad += momentum * m_buffer
                else:
                    grad = m_buffer

                original_shape = grad.shape
                if grad.ndim == 4:
                    grad = grad.reshape(original_shape[0], -1)
                elif grad.ndim != 2:
                    raise ValueError(f"param ndim {grad.ndim} not supported")

                grad = self._newton_schulz(grad, ns_coefficients, eps, ns_steps)

                if len(original_shape) == 4:
                    grad = grad.reshape(original_shape)

                param._requires_grad = False
                np.multiply(param.to_np(), 1 - lr * weight_decay, out=param.to_np())
                lr_ = adjust_lr_fn(
                    lr, original_shape[0], grad.size // original_shape[0]
                )
                np.subtract(param.to_np(), lr_ * grad, out=param.to_np())
                param._requires_grad = True
