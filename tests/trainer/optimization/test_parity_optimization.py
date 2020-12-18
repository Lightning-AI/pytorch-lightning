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
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.optim import Optimizer

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.base.boring_model import BoringModel

"""
TODO:
For both automatic / manual optimization
    - Test dp, ddp, ddp2
    - Apex
    - Random accumulated_grad_batches (bug)
    - Multiple optimizers
"""


@pytest.mark.parametrize(["precision", "amp_backend", "gpus"], [
    pytest.param(32, "native", 0),
    pytest.param(16, "native", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires GPU')),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1])
def test_lightning_optimizer_and_no_lightning_optimizer_equality(
        tmpdir,
        precision,
        amp_backend,
        gpus,
        accumulate_grad_batches):
    run_pl_optimizer_equality(
        precision=precision,
        default_root_dir=tmpdir,
        accumulate_grad_batches=accumulate_grad_batches,
        amp_backend=amp_backend,
        gpus=gpus
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
        accumulate_grad_batches):
    with patch("torch.optim.SGD.step") as mock_sdg_step, \
            patch("torch.optim.Adam.step") as mock_adam_step, \
            patch("torch.optim.AdamW.step") as mock_adamw_step, \
            patch("torch.optim.SGD.zero_grad") as mock_sdg_zero_grad, \
            patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad, \
            patch("torch.optim.AdamW.zero_grad") as mock_adamw_zero_grad:
        expected_num_batches = run_pl_optimizer_equality(
            default_root_dir=tmpdir,
            optimizer_is_mocked=True,
            accumulate_grad_batches=accumulate_grad_batches,
            amp_backend=amp_backend,
            precision=precision,
            gpus=gpus
        )

        assert mock_sdg_step.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sdg_zero_grad.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_zero_grad.call_count == mock_adam_zero_grad.call_count
        assert mock_sdg_zero_grad.call_count == mock_adamw_zero_grad.call_count


def run_pl_optimizer_equality(optimizer_is_mocked=False, **train_kwargs):
    trainer_kwargs = {
        "max_epochs": np.random.randint(1, 3),
        "limit_train_batches": np.random.randint(11, 27),
        "limit_val_batches": 0,
        **train_kwargs
    }
    expected_num_batches = trainer_kwargs["max_epochs"] * trainer_kwargs["limit_train_batches"]
    accumulate_grad_batches = trainer_kwargs["accumulate_grad_batches"]
    expected_global_step = expected_num_batches // accumulate_grad_batches

    seed_everything(42)
    model = BaseParityAutomaticOptimizationModel(
        "Adam" if optimizer_is_mocked else "SGD",
        optimizer_is_mocked=optimizer_is_mocked
    )

    initial_model_weights_pl_optimizer, model_pl_optimizer = train_model(
        model=model,
        expected_global_step=expected_global_step,
        enable_pl_optimizer=True,
        **trainer_kwargs,
    )

    seed_everything(42)
    model = BaseParityAutomaticOptimizationModel(
        "AdamW" if optimizer_is_mocked else "SGD",
        optimizer_is_mocked=optimizer_is_mocked
    )

    initial_model_weights_no_pl_optimizer, model_no_pl_optimizer = train_model(
        model=model,
        expected_global_step=expected_global_step,
        enable_pl_optimizer=False,  # Disable pl optimizer
        **trainer_kwargs,
    )
    assert_model_equality(
        initial_model_weights_pl_optimizer=initial_model_weights_pl_optimizer,
        model_pl_optimizer=model_pl_optimizer,
        initial_model_weights_no_pl_optimizer=initial_model_weights_no_pl_optimizer,
        model_no_pl_optimizer=model_no_pl_optimizer,
        expected_num_batches=expected_num_batches
    )
    return expected_num_batches


def assert_model_equality(
        initial_model_weights_pl_optimizer,
        model_pl_optimizer,
        initial_model_weights_no_pl_optimizer,
        model_no_pl_optimizer,
        expected_num_batches):
    assert torch.equal(initial_model_weights_pl_optimizer, initial_model_weights_no_pl_optimizer)
    assert len(model_pl_optimizer.losses) == expected_num_batches

    assert not torch.FloatTensor(model_no_pl_optimizer.losses).isnan().any()

    assert torch.equal(torch.FloatTensor(model_no_pl_optimizer.losses), torch.FloatTensor(model_pl_optimizer.losses))
    assert model_no_pl_optimizer.on_before_zero_grad_count == model_pl_optimizer.on_before_zero_grad_count

    for b_grad, a_grad in zip(model_no_pl_optimizer.grads, model_pl_optimizer.grads):
        assert torch.equal(b_grad, a_grad), 'Grad parameters are different'

    for b_w, a_w in zip(model_no_pl_optimizer.parameters(), model_pl_optimizer.parameters()):
        assert torch.equal(b_w, a_w), 'Model parameters are different'


# train function
def train_model(model, expected_global_step, **trainer_kwargs):
    initial_weights = model.layer.weight.clone()
    model.training_epoch_end = None

    trainer = Trainer(
        **trainer_kwargs
    )
    trainer.fit(model)

    assert np.abs(trainer.global_step - expected_global_step) <= 2
    return initial_weights, model


class BaseParityAutomaticOptimizationModel(BoringModel):

    def __init__(self, optimizer_name, optimizer_is_mocked=False):
        super().__init__()
        self._optimizer_name = optimizer_name
        self.losses = []
        self.grads = []
        self.on_before_zero_grad_count = 0
        self.optimizer_is_mocked = optimizer_is_mocked
        self.grad_checked = False

    def on_before_zero_grad(self, optimizer):
        self.on_before_zero_grad_count += 1
        if self.layer.weight.grad is not None:
            self.grads.append(self.layer.weight.grad.clone())

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self._optimizer_name)(self.layer.parameters(), lr=0.1)
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        return {"loss": loss}


class AutomaticOptimizationNativeModel(BaseParityAutomaticOptimizationModel):

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:
        """
        Override the optimizer step to define manual optimizer steps, as we use LightningOptimizer wrapper as standard
        """
        # Get the unwrapped optimizer
        optimizer = optimizer._optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        optimizer_closure()
        if should_accumulate(self.trainer):
            return

        self.grad_checked = True
        assert torch.abs(self.layer.weight.grad).sum() > 0
        self.on_before_zero_grad(optimizer)
        optimizer.step()
        optimizer.zero_grad()

        if not self.optimizer_is_mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


class AutomaticOptimizationVanillaAMPNativeModel(BaseParityAutomaticOptimizationModel):

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:
        """
        Override the optimizer step to define manual optimizer steps, as we use LightningOptimizer wrapper as standard
        """
        # Get the unwrapped optimizer
        optimizer = optimizer._optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        optimizer_closure()
        if should_accumulate(self.trainer):
            return

        self.grad_checked = True
        assert 0 < torch.abs(self.layer.weight.grad).sum()
        self.on_before_zero_grad(optimizer)
        self.trainer.scaler.step(optimizer)

        if self.optimizer_is_mocked:
            optimizer.step()

        self.trainer.scaler.update()
        optimizer.zero_grad()

        if not self.optimizer_is_mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


def should_accumulate(trainer):
    accumulation_done = (trainer.batch_idx + 1) == trainer.num_training_batches
    is_final_batch = (trainer.batch_idx + 1) % trainer.accumulate_grad_batches == 0
    return not (accumulation_done or is_final_batch)
