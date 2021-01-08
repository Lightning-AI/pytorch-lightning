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

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.base.boring_model import BoringModel

# TODO:
# For both automatic / manual optimization
#     - Test dp, ddp, ddp2
#     - Apex
#     - Random accumulated_grad_batches (bug)
#     - Multiple optimizers


class BaseParityAutomaticOptimizationModel(BoringModel):

    def __init__(
        self,
        optimizer_cls,
        optimizer_is_mocked=False,
        accumulate_grad_batches=None,
        lr=0.1
    ):
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.losses = []
        self.grads = []
        self.on_before_zero_grad_count = 0
        self.optimizer_is_mocked = optimizer_is_mocked
        self.grad_checked = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.lr = lr

    def on_before_zero_grad(self, optimizer):
        self.on_before_zero_grad_count += 1
        if self.layer.weight.grad is not None:
            self.grads.append(self.layer.weight.grad.clone())

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.layer.parameters(), lr=self.lr)
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        return {"loss": loss}


class AutomaticOptimizationPurePytorchOptimizerModel(BaseParityAutomaticOptimizationModel):

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.losses.append(loss.detach().item())
        loss /= float(self.accumulate_grad_batches)
        return {"loss": loss}

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
        optimizer = optimizer.optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        optimizer_closure()
        assert self.trainer.accumulate_grad_batches == 1

        if should_accumulate(self.trainer, self.accumulate_grad_batches):
            return

        self.grad_checked = True
        assert torch.abs(self.layer.weight.grad).sum() > 0
        optimizer.step()

        self.on_before_zero_grad_count += 1
        optimizer.zero_grad()

        if not self.optimizer_is_mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


class AutomaticOptimizationPurePytorchAMPOptimizerModel(BaseParityAutomaticOptimizationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.cuda.amp.GradScaler()

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.losses.append(loss.detach().item())
            loss /= float(self.accumulate_grad_batches)
            loss = self.scaler.scale(loss)
            return {"loss": loss}

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
        optimizer = optimizer.optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        optimizer_closure()
        assert self.trainer.accumulate_grad_batches == 1

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


def should_accumulate(trainer, accumulate_grad_batches):
    accumulation_done = (trainer.batch_idx + 1) == trainer.num_training_batches
    is_final_batch = (trainer.batch_idx + 1) % accumulate_grad_batches == 0
    return not (accumulation_done or is_final_batch)


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
        accumulate_grad_batches,
):

    if accumulate_grad_batches > 1:
        accumulate_grad_batches = np.random.randint(1, accumulate_grad_batches)

    vanilla_model_cls = AutomaticOptimizationPurePytorchAMPOptimizerModel if precision == 16 \
        else AutomaticOptimizationPurePytorchOptimizerModel

    run_lightning_optimizer_equality(
        BaseParityAutomaticOptimizationModel,
        vanilla_model_cls,
        precision=precision,
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
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
        accumulate_grad_batches,
):

    vanilla_model_cls = AutomaticOptimizationPurePytorchAMPOptimizerModel if precision == 16 \
        else AutomaticOptimizationPurePytorchOptimizerModel

    with patch("torch.optim.SGD.step") as mock_sgd_step, \
            patch("torch.optim.Adam.step") as mock_adam_step, \
            patch("torch.optim.SGD.zero_grad") as mock_sgd_zero_grad, \
            patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad:

        max_epochs = 2
        limit_train_batches = 10

        # Run equality test using Lightning Optimizer
        run_lightning_optimizer_equality(
            BaseParityAutomaticOptimizationModel,
            vanilla_model_cls,
            default_root_dir=tmpdir,
            optimizer_is_mocked=True,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            amp_backend=amp_backend,
            precision=precision,
            gpus=gpus
        )

        expected_num_batches = max_epochs * limit_train_batches
        assert mock_sgd_step.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_zero_grad.call_count == (expected_num_batches // accumulate_grad_batches)
        assert mock_sgd_step.call_count == mock_adam_step.call_count
        assert mock_sgd_step.call_count == mock_adam_step.call_count


def train_with_restore(tmpdir, model_cls, restore_from=None):
    # init model
    if restore_from is not None:
        seed_everything(42)
    model = model_cls(torch.optim.Adam, accumulate_grad_batches=1, lr=10e-1)
    ckpt_saver = ModelCheckpoint(dirpath=f"{tmpdir}/mckpt", save_last=True, save_weights_only=False)
    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=(1 + bool(restore_from)),
        limit_train_batches=8,
        callbacks=([ckpt_saver] if restore_from is None else []),
        checkpoint_callback=(not restore_from),
        resume_from_checkpoint=restore_from,
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model)
    return ckpt_saver.best_model_path, model


def test_parity_checkpointing(tmpdir):
    """
    This test assert that reloading a checkpoint and finetunning gives the same result
    with / without LightningOptimizer
    """

    # Initial train run of the model.
    seed_everything(0)
    ckpt_path, first_epoch_pl_optimizer_model = train_with_restore(
        tmpdir,
        model_cls=BaseParityAutomaticOptimizationModel,
        restore_from=None)

    assert "last" in ckpt_path
    _, second_epoch_pl_optimizer_model = train_with_restore(
        tmpdir,
        model_cls=BaseParityAutomaticOptimizationModel,
        restore_from=ckpt_path)

    seed_everything(0)
    ckpt_path, first_epoch_pure_pytorch_optimizer_model = train_with_restore(
        tmpdir,
        model_cls=AutomaticOptimizationPurePytorchOptimizerModel,
        restore_from=None)

    _, second_epoch_pure_pytorch_optimizer_model = train_with_restore(
        tmpdir,
        model_cls=AutomaticOptimizationPurePytorchOptimizerModel,
        restore_from=ckpt_path)

    assert first_epoch_pl_optimizer_model.losses == first_epoch_pure_pytorch_optimizer_model.losses
    assert second_epoch_pl_optimizer_model.losses == second_epoch_pure_pytorch_optimizer_model.losses


def run_lightning_optimizer_equality(
        lightning_model_cls,
        vanilla_model_cls,
        optimizer_is_mocked=False,
        **trainer_kwargs,
):

    trainer_kwargs = {
        "limit_val_batches": 0,
        **trainer_kwargs
    }
    expected_num_batches = trainer_kwargs["max_epochs"] * trainer_kwargs["limit_train_batches"]
    accumulate_grad_batches = trainer_kwargs["accumulate_grad_batches"]

    pl_optimizer_initial_model_weights, pl_optimizer_model = train_specific_optimizer_model(
        lightning_model_cls,
        torch.optim.SGD,
        expected_num_batches=expected_num_batches,
        optimizer_is_mocked=optimizer_is_mocked,
        **trainer_kwargs,
    )

    pure_pytorch_optimizer_initial_model_weights, pure_pytorch_optimizer_model = train_specific_optimizer_model(
        vanilla_model_cls,
        torch.optim.Adam if optimizer_is_mocked else torch.optim.SGD,
        expected_num_batches=expected_num_batches,
        optimizer_is_mocked=optimizer_is_mocked,
        replace_optimizer_step_with_pure_pytorch=True,
        **trainer_kwargs,
    )

    if not optimizer_is_mocked:

        assert_model_equality(
            pl_optimizer_initial_model_weights=pl_optimizer_initial_model_weights,
            pl_optimizer_model=pl_optimizer_model,
            pure_pytorch_optimizer_initial_model_weights=pure_pytorch_optimizer_initial_model_weights,
            pure_pytorch_optimizer_model=pure_pytorch_optimizer_model,
            expected_num_batches=expected_num_batches,
            precision=trainer_kwargs["precision"]
        )


def assert_model_equality(
        pl_optimizer_initial_model_weights,
        pl_optimizer_model,
        pure_pytorch_optimizer_initial_model_weights,
        pure_pytorch_optimizer_model,
        expected_num_batches,
        precision,
):

    assert torch.equal(pl_optimizer_initial_model_weights, pure_pytorch_optimizer_initial_model_weights)
    assert len(pl_optimizer_model.losses) == expected_num_batches
    assert pure_pytorch_optimizer_model.grad_checked
    assert not torch.isnan(torch.FloatTensor(pl_optimizer_model.losses)).any()

    for pytorch_grad, pl_optim_grad in zip(pure_pytorch_optimizer_model.grads,
                                           pl_optimizer_model.grads):
        assert torch.equal(pytorch_grad, pl_optim_grad), 'Grad parameters are different'

    for pytorch_weight, pl_optim_weight in zip(pure_pytorch_optimizer_model.parameters(),
                                               pl_optimizer_model.parameters()):
        assert torch.equal(pytorch_weight, pl_optim_weight), 'Model parameters are different'


# train function
def train_specific_optimizer_model(
        model_cls,
        optimizer_cls,
        expected_num_batches,
        optimizer_is_mocked=False,
        replace_optimizer_step_with_pure_pytorch=False,
        **trainer_kwargs,
):

    seed_everything(42)
    trainer_kwargs = deepcopy(trainer_kwargs)

    model = model_cls(
        optimizer_cls=optimizer_cls,
        optimizer_is_mocked=optimizer_is_mocked,
        accumulate_grad_batches=trainer_kwargs["accumulate_grad_batches"],
    )

    if replace_optimizer_step_with_pure_pytorch:
        # When running pure vanilla training, accumulate_grad_batches should be 1.
        trainer_kwargs["accumulate_grad_batches"] = 1
        trainer_kwargs["precision"] = 32

    expected_global_step = expected_num_batches // trainer_kwargs["accumulate_grad_batches"]

    initial_weights = model.layer.weight.clone()
    model.training_epoch_end = None

    trainer = Trainer(
        **trainer_kwargs
    )
    trainer.fit(model)

    assert np.abs(trainer.global_step - expected_global_step) <= 2
    return initial_weights, model
