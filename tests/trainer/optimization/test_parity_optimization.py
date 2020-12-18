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
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_utils import is_overridden
from tests.base.boring_model import BoringModel, RandomDataset, RandomDictDataset, RandomDictStringDataset
from tests.base.models import BasicGAN

"""
TODO:
For both automatic / manual optimization
    - Test dp, ddp, ddp2
    - Apex
    - Random accumulated_grad_batches (bug)
    - Multiple optimizers
"""

##################################################
# TESTING AUTOMATIC OPTIMIZATION - ONE OPTIMIZER #
##################################################

################## MODULES ##################   # noqa E266


class BaseParityAutomaticOptimizationModel(BoringModel):

    def __init__(self, optimizer_name, mocked=False):
        super().__init__()
        self._optimizer_name = optimizer_name
        self.losses = []
        self.grads = []
        self.on_before_zero_grad_count = 0
        self.mocked = mocked
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


class AutomatiocOptimizationNativeModel(BaseParityAutomaticOptimizationModel):

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Getting the un-wrapped optimizer
        optimizer = optimizer._optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        closure()
        if should_accumulate(self.trainer):
            return

        self.grad_checked = True
        assert torch.abs(self.layer.weight.grad).sum() > 0
        self.on_before_zero_grad(optimizer)
        optimizer.step()
        optimizer.zero_grad()

        if not self.mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0


class AutomatiocOptimizationVanillaAMPNativeModel(BaseParityAutomaticOptimizationModel):

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Getting the un-wrapped optimizer
        optimizer = optimizer._optimizer
        assert not isinstance(optimizer, LightningOptimizer)

        closure()
        if should_accumulate(self.trainer):
            return

        self.grad_checked = True
        assert 0 < torch.abs(self.layer.weight.grad).sum()
        self.on_before_zero_grad(optimizer)
        self.trainer.scaler.step(optimizer)

        if self.mocked:
            optimizer.step()

        self.trainer.scaler.update()
        optimizer.zero_grad()

        if not self.mocked:
            assert torch.abs(self.layer.weight.grad).sum() == 0

################## HELPERS ##################   # noqa E266


def should_accumulate(trainer):
    accumulation_done = (trainer.batch_idx + 1) == trainer.num_training_batches
    is_final_batch = (trainer.batch_idx + 1) % trainer.accumulate_grad_batches == 0
    return not (accumulation_done or is_final_batch)


# train function
def parity_automatic_train_with_one_optimizer(ctx):
    seed_everything(42)
    expected_batches = ctx["expected_batches"]
    accumulate_grad_batches = ctx["accumulate_grad_batches"]
    if ctx["vanilla"]:
        # Note: global_step counts training_loop.optimizer_step
        expected_global_step = expected_batches
        if ctx["using_amp"]:
            model = AutomatiocOptimizationVanillaAMPNativeModel("SGD", mocked=ctx["mocked"])
        else:
            model = AutomatiocOptimizationNativeModel("SGD", mocked=ctx["mocked"])
        ctx["initial_weights"]["vanilla_model"] = model.layer.weight.clone()
    else:
        expected_global_step = (expected_batches) // accumulate_grad_batches
        if ctx["enable_pl_optimizer"]:
            model = BaseParityAutomaticOptimizationModel("Adam" if ctx["mocked"] else "SGD", mocked=ctx["mocked"])
            ctx["initial_weights"]["model_wi_pl_optimizer"] = model.layer.weight.clone()
        else:
            model = BaseParityAutomaticOptimizationModel("AdamW" if ctx["mocked"] else "SGD", mocked=ctx["mocked"])
            ctx["initial_weights"]["model_wo_pl_optimizer"] = model.layer.weight.clone()

    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=ctx["tmpdir"],
        max_epochs=ctx["max_epochs"],
        limit_train_batches=ctx["limit_train_batches"],
        limit_val_batches=ctx["limit_val_batches"],
        enable_pl_optimizer=ctx["enable_pl_optimizer"],
        accumulate_grad_batches=accumulate_grad_batches if not ctx["vanilla"] else 1,
        amp_backend=ctx["amp_backend"],
        precision=ctx["precision"],
        gpus=1
    )
    trainer.fit(model)

    assert np.abs(trainer.global_step - expected_global_step) <= 2
    return model


################## TESTS ##################     # noqa E266


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(["precision", "amp_backend"], [
    pytest.param(16, "native"),
    pytest.param(32, "native"),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1, 2])
def test_parity_automatic_training_with_one_optimizer(tmpdir, amp_backend, precision, accumulate_grad_batches):
    """
    Test training with accumulated gradients with and within enable_pl_optimizer reaches the same weights
    """
    # prepare arguments
    if accumulate_grad_batches > 1:
        accumulate_grad_batches = np.random.randint(2, accumulate_grad_batches + 1)

    ctx = {}
    ctx["tmpdir"] = tmpdir
    ctx["accumulate_grad_batches"] = accumulate_grad_batches
    ctx["amp_backend"] = amp_backend
    ctx["precision"] = precision
    ctx["using_amp"] = (amp_backend in ["native"]) and precision == 16
    ctx["max_epochs"] = np.random.randint(1, 3)
    ctx["limit_train_batches"] = np.random.randint(11, 27)
    expected_batches = ctx["max_epochs"] * ctx["limit_train_batches"]
    ctx["expected_batches"] = expected_batches
    ctx["limit_val_batches"] = 0
    ctx["initial_weights"] = {}
    ctx["enable_pl_optimizer"] = True
    ctx["mocked"] = False
    ctx["vanilla"] = False

    model_wi_pl_optimizer = parity_automatic_train_with_one_optimizer(ctx)

    ctx["enable_pl_optimizer"] = False
    model_wo_pl_optimizer = parity_automatic_train_with_one_optimizer(ctx)

    # assertions
    assert torch.equal(ctx["initial_weights"]["model_wo_pl_optimizer"], ctx["initial_weights"]["model_wi_pl_optimizer"])
    assert len(model_wo_pl_optimizer.losses) == expected_batches

    assert np.abs(len(model_wo_pl_optimizer.grads) - (expected_batches // accumulate_grad_batches)) <= 2
    assert not torch.FloatTensor(model_wo_pl_optimizer.losses).isnan().any()

    assert torch.equal(torch.FloatTensor(model_wo_pl_optimizer.losses), torch.FloatTensor(model_wi_pl_optimizer.losses))
    assert model_wo_pl_optimizer.on_before_zero_grad_count == model_wi_pl_optimizer.on_before_zero_grad_count

    for b_grad, a_grad in zip(model_wo_pl_optimizer.grads, model_wi_pl_optimizer.grads):
        assert torch.equal(b_grad, a_grad), 'Grad parameters are different'

    for b_w, a_w in zip(model_wo_pl_optimizer.parameters(), model_wi_pl_optimizer.parameters()):
        assert torch.equal(b_w, a_w), 'Model parameters are different'

    ctx["vanilla"] = True
    vanilla_model = parity_automatic_train_with_one_optimizer(ctx)

    assert torch.equal(ctx["initial_weights"]["model_wo_pl_optimizer"], ctx["initial_weights"]["vanilla_model"])
    assert vanilla_model.grad_checked
    assert vanilla_model.losses == model_wo_pl_optimizer.losses
    assert (vanilla_model.on_before_zero_grad_count // accumulate_grad_batches) == model_wo_pl_optimizer.on_before_zero_grad_count

    for b_grad, o_grad in zip(model_wo_pl_optimizer.grads, vanilla_model.grads):
        assert torch.equal(b_grad, o_grad), 'Grad parameters are different'

    for b_w, o_w in zip(model_wo_pl_optimizer.parameters(), vanilla_model.parameters()):
        assert torch.equal(b_w, o_w), 'Model parameters are different'

    # 16-bit with AMP Native needs to adapted as it uses trainer.scaler.step(optimizer)
    if precision == 32:

        with patch("torch.optim.SGD.step") as mock_sdg_step, \
             patch("torch.optim.Adam.step") as mock_adam_step, \
             patch("torch.optim.AdamW.step") as mock_adamw_step, \
             patch("torch.optim.SGD.zero_grad") as mock_sdg_zero_grad, \
             patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad, \
             patch("torch.optim.AdamW.zero_grad") as mock_adamw_zero_grad:

            ctx["mocked"] = True
            parity_automatic_train_with_one_optimizer(ctx)
            ctx["vanilla"] = False
            parity_automatic_train_with_one_optimizer(ctx)
            ctx["enable_pl_optimizer"] = True
            parity_automatic_train_with_one_optimizer(ctx)

        assert mock_sdg_step.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_zero_grad.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_zero_grad.call_count == mock_adam_zero_grad.call_count
        assert mock_sdg_zero_grad.call_count == mock_adamw_zero_grad.call_count

###############################################
# TESTING MANUAL OPTIMIZATION - ONE OPTIMIZER #
###############################################


########################################################
# TESTING AUTOMATIC OPTIMIZATION - MULTIPLE OPTIMIZERS #
########################################################


####################################################
# TESTING MANUAL OPTIMIZATION - MULTIPLE OPTIMIZER #
####################################################
