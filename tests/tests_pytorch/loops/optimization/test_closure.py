# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Callable, Iterable, Tuple

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_optimizer_step_no_closure_raises(tmpdir):
    class TestModel(BoringModel):
        def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None, **_):
            # does not call `optimizer_closure()`
            pass

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            class BrokenSGD(torch.optim.SGD):
                def step(self, closure=None):
                    # forgot to pass the closure
                    return super().step()

            return BrokenSGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)


def test_closure_with_no_grad_optimizer(tmpdir):
    """
    The reason for conducting this test is because there are certain
    third-party library optimizers (such as Hugging Face Transformers' AdamW)
    that set `no_grad` during the `step` operation.

    Lightning wraps hooks like `training_step` and `backward` within a closure,
    which is executed during `optimizer.step`.

    This can lead to a specific issue where the gradient is missing
    for the `training_step` loss.
    """

    class AdamW(Optimizer):
        """
        The AdamW optimizer from Hugging Face transformers library:
        https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
        """

        def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            no_deprecation_warning: bool = False,
        ):
            if lr < 0.0:
                raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
            if not eps >= 0.0:
                raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
            defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "correct_bias": correct_bias,
            }
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure: Callable = None):
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

            return loss

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return AdamW(self.parameters(), lr=0.1)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    model = TestModel()
    trainer.fit(model)
