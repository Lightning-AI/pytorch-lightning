# Copyright The PyTorch Lightning team.
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
from collections import Callable
from copy import deepcopy
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.optim import Optimizer

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.base.boring_model import BoringModel
from tests.trainer.optimization.test_parity_automatic_optimization import (
    assert_model_equality,
    run_lightning_optimizer_equality,
    should_accumulate,
)

"""
TODO:
For both Manual / manual optimization
    - Test dp, ddp, ddp2
    - Apex
    - Random accumulated_grad_batches (bug)
    - Multiple optimizers
"""


class BaseParityManualOptimizationModel(BoringModel):

    def __init__(self, optimizer_cls, optimizer_is_mocked=False, accumulate_grad_batches=None):
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.losses = []
        self.grads = []
        self.on_before_zero_grad_count = 0
        self.optimizer_is_mocked = optimizer_is_mocked
        self.grad_checked = False
        self.accumulate_grad_batches = accumulate_grad_batches

    def on_before_zero_grad(self, optimizer):
        self.on_before_zero_grad_count += 1
        if self.layer.weight.grad is not None:
            self.grads.append(self.layer.weight.grad.clone())

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.layer.parameters(), lr=0.1)
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        if not isinstance(opt, LightningOptimizer):
            opt = LightningOptimizer.to_lightning_optimizer(opt, self.trainer)
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        self.manual_backward(loss, opt)
        opt.step()


class ManualOptimizationPurePytorchOptimizerModel(BaseParityManualOptimizationModel):

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=False)
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        loss /= float(self.accumulate_grad_batches)
        loss.backward()

        if should_accumulate(self.trainer, self.accumulate_grad_batches):
            return

        self.grad_checked = True
        assert torch.abs(self.layer.weight.grad).sum() > 0
        optimizer.step()

        self.on_before_zero_grad_count += 1
        optimizer.zero_grad()

        if not self.optimizer_is_mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


class ManualOptimizationPurePytorchAMPOptimizerModel(BaseParityManualOptimizationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.cuda.amp.GradScaler()

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=False)
        with torch.cuda.amp.autocast():
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.losses.append(loss.detach().item())
            loss /= float(self.accumulate_grad_batches)
            loss = self.scaler.scale(loss)
            loss.backward()

        if should_accumulate(self.trainer, self.accumulate_grad_batches):
            return

        self.scaler.unscale_(optimizer)
        self.grad_checked = True

        assert torch.abs(self.layer.weight.grad).sum() > 0
        self.scaler.step(optimizer)
        self.scaler.update()
        self.on_before_zero_grad_count += 1
        optimizer.zero_grad()

        if not self.optimizer_is_mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


@pytest.mark.parametrize(["precision", "amp_backend", "gpus"], [
    pytest.param(32, "native", 0),
    pytest.param(16, "native", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires GPU')),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1, 7])
def test_lightning_optimizer_and_no_lightning_optimizer_equality(
        tmpdir,
        precision,
        amp_backend,
        gpus,
        accumulate_grad_batches):

    if accumulate_grad_batches > 1:
        accumulate_grad_batches = np.random.randint(1, accumulate_grad_batches)

    vanilla_model_cls = ManualOptimizationPurePytorchAMPOptimizerModel if precision == 16 \
        else ManualOptimizationPurePytorchOptimizerModel

    run_lightning_optimizer_equality(
        BaseParityManualOptimizationModel,
        vanilla_model_cls,
        precision=precision,
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        accumulate_grad_batches=accumulate_grad_batches,
        amp_backend=amp_backend,
        gpus=gpus,
        automatic_optimization=False
    )


@pytest.mark.parametrize(["precision", "amp_backend", "gpus"], [
    pytest.param(32, "native", 0),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1])
def test_lightning_optimizer_and_no_lightning_optimizer_equality_check_optim_calls(
        tmpdir,
        precision,
        amp_backend,
        gpus,
        accumulate_grad_batches,
):

    vanilla_model_cls = ManualOptimizationPurePytorchAMPOptimizerModel if precision == 16 \
        else ManualOptimizationPurePytorchOptimizerModel

    with patch("torch.optim.SGD.step") as mock_sgd_step, \
            patch("torch.optim.Adam.step") as mock_adam_step, \
            patch("torch.optim.SGD.zero_grad") as mock_sgd_zero_grad, \
            patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad:

        max_epochs = 2
        limit_train_batches = 10

        # Run equality test using Lightning Optimizer

        run_lightning_optimizer_equality(
            BaseParityManualOptimizationModel,
            vanilla_model_cls,
            default_root_dir=tmpdir,
            optimizer_is_mocked=True,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            amp_backend=amp_backend,
            precision=precision,
            gpus=gpus,
            automatic_optimization=False
        )

        expected_num_batches = max_epochs * limit_train_batches
        assert mock_sgd_step.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_zero_grad.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_step.call_count == mock_adam_step.call_count
        assert mock_sgd_zero_grad.call_count == mock_adam_zero_grad.call_count
