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
    run_lightning_optimizer_equality(
        precision=precision,
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=10,
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
    with patch("torch.optim.SGD.step") as mock_sgd_step, \
            patch("torch.optim.Adam.step") as mock_adam_step, \
            patch("torch.optim.AdamW.step") as mock_adamw_step, \
            patch("torch.optim.SGD.zero_grad") as mock_sgd_zero_grad, \
            patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad, \
            patch("torch.optim.AdamW.zero_grad") as mock_adamw_zero_grad:
        max_epochs = 2
        limit_train_batches = 10

        # Run equality test using Lightning Optimizer
        run_lightning_optimizer_equality(
            default_root_dir=tmpdir,
            optimizer_is_mocked=True,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=2,
            limit_train_batches=10,
            amp_backend=amp_backend,
            precision=precision,
            gpus=gpus
        )

        expected_num_batches = max_epochs * limit_train_batches
        assert mock_sgd_step.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_zero_grad.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_step.call_count == mock_adam_step.call_count
        assert mock_sgd_step.call_count == mock_adam_step.call_count
        assert mock_sgd_zero_grad.call_count == mock_adam_zero_grad.call_count
        assert mock_sgd_zero_grad.call_count == mock_adamw_zero_grad.call_count


def run_lightning_optimizer_equality(
        optimizer_is_mocked=False,
        **train_kwargs):
    trainer_kwargs = {
        "limit_val_batches": 0,
        **train_kwargs
    }
    expected_num_batches = trainer_kwargs["max_epochs"] * trainer_kwargs["limit_train_batches"]
    accumulate_grad_batches = trainer_kwargs["accumulate_grad_batches"]
    expected_global_step = expected_num_batches // accumulate_grad_batches

    pl_optimizer_initial_model_weights, pl_optimizer_model = train_specific_optimizer_model(
        expected_global_step=expected_global_step,
        optimizer_is_mocked=optimizer_is_mocked,
        enable_pl_optimizer=True,
        **trainer_kwargs,
    )

    no_pl_optimizer_initial_model_weights, no_pl_optimizer_model = train_specific_optimizer_model(
        expected_global_step=expected_global_step,
        optimizer_is_mocked=optimizer_is_mocked,
        enable_pl_optimizer=False,  # Disable pl optimizer
        **trainer_kwargs,
    )

    pure_pytorch_optimizer_initial_model_weights, pure_pytorch_optimizer_model = train_specific_optimizer_model(
        expected_global_step=expected_global_step,
        optimizer_is_mocked=optimizer_is_mocked,
        replace_optimizer_step_with_pure_pytorch=True,
        **trainer_kwargs,
    )

    assert_model_equality(
        pl_optimizer_initial_model_weights=pl_optimizer_initial_model_weights,
        pl_optimizer_model=pl_optimizer_model,
        no_pl_optimizer_initial_model_weights=no_pl_optimizer_initial_model_weights,
        no_pl_optimizer_model=no_pl_optimizer_model,
        pure_pytorch_optimizer_initial_model_weights=pure_pytorch_optimizer_initial_model_weights,
        pure_pytorch_optimizer_model=pure_pytorch_optimizer_model,
        expected_num_batches=expected_num_batches
    )


def assert_model_equality(
        pl_optimizer_initial_model_weights,
        pl_optimizer_model,
        no_pl_optimizer_initial_model_weights,
        no_pl_optimizer_model,
        pure_pytorch_optimizer_initial_model_weights,
        pure_pytorch_optimizer_model,
        expected_num_batches):
    assert torch.equal(pl_optimizer_initial_model_weights, no_pl_optimizer_initial_model_weights)
    assert torch.equal(pl_optimizer_initial_model_weights, pure_pytorch_optimizer_initial_model_weights)
    assert len(pl_optimizer_model.losses) == expected_num_batches
    assert pure_pytorch_optimizer_model.grad_checked
    assert pure_pytorch_optimizer_model.losses == no_pl_optimizer_model.losses
    assert not torch.FloatTensor(no_pl_optimizer_model.losses).isnan().any()

    assert torch.equal(torch.FloatTensor(no_pl_optimizer_model.losses), torch.FloatTensor(pl_optimizer_model.losses))
    assert no_pl_optimizer_model.on_before_zero_grad_count == pl_optimizer_model.on_before_zero_grad_count

    for pytorch_grad, no_pl_optim_grad, pl_optim_grad in zip(pure_pytorch_optimizer_model.parameters(),
                                                             no_pl_optimizer_model.parameters(),
                                                             pl_optimizer_model.parameters()):
        assert torch.equal(no_pl_optim_grad, pl_optim_grad), 'Grad parameters are different'
        assert torch.equal(pytorch_grad, no_pl_optim_grad), 'Grad parameters are different'

    for pytorch_weight, no_pl_optim_weight, pl_optim_weight in zip(pure_pytorch_optimizer_model.parameters(),
                                                                   no_pl_optimizer_model.parameters(),
                                                                   pl_optimizer_model.parameters()):
        assert torch.equal(no_pl_optim_weight, pl_optim_weight), 'Model parameters are different'
        assert torch.equal(pytorch_weight, no_pl_optim_weight), 'Model parameters are different'


# train function
def train_specific_optimizer_model(
        expected_global_step,
        enable_pl_optimizer=False,
        optimizer_is_mocked=False,
        replace_optimizer_step_with_pure_pytorch=False,
        **trainer_kwargs):
    seed_everything(42)

    if trainer_kwargs["precision"] == 16 and replace_optimizer_step_with_pure_pytorch:
        model_cls = AutomaticOptimizationPurePytorchAMPOptimizerModel
    else:
        model_cls = AutomaticOptimizationPurePytorchOptimizerModel if replace_optimizer_step_with_pure_pytorch \
            else BaseParityAutomaticOptimizationModel
    model = model_cls(
        optimizer_cls=torch.optim.AdamW if optimizer_is_mocked else torch.optim.SGD,
        optimizer_is_mocked=optimizer_is_mocked
    )

    initial_weights = model.layer.weight.clone()
    model.training_epoch_end = None

    trainer = Trainer(
        enable_pl_optimizer=enable_pl_optimizer,
        **trainer_kwargs
    )
    trainer.fit(model)

    assert np.abs(trainer.global_step - expected_global_step) <= 2
    return initial_weights, model


class BaseParityAutomaticOptimizationModel(BoringModel):

    def __init__(self, optimizer_cls, optimizer_is_mocked=False):
        super().__init__()
        self.optimizer_cls = optimizer_cls
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
        optimizer = self.optimizer_cls(self.layer.parameters(), lr=0.1)
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        return {"loss": loss}


class AutomaticOptimizationPurePytorchOptimizerModel(BaseParityAutomaticOptimizationModel):

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


class AutomaticOptimizationPurePytorchAMPOptimizerModel(BaseParityAutomaticOptimizationModel):

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
