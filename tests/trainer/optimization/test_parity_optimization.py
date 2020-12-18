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


################## TESTS ##################     # noqa E266


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(["precision", "amp_backend"], [
    pytest.param(16, "native"),
    pytest.param(32, "native"),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1])
def test_parity_training_lightning_optimizer_one_optimizer(tmpdir, amp_backend, precision, accumulate_grad_batches):
    """
    Test training with accumulated gradients with and within enable_pl_optimizer reaches the same weights
    """
    # prepare arguments
    if accumulate_grad_batches > 1:
        accumulate_grad_batches = np.random.randint(2, accumulate_grad_batches + 1)

    using_amp = (amp_backend in ["native"]) and precision == 16
    max_epochs = np.random.randint(1, 3)
    limit_train_batches = np.random.randint(11, 27)
    expected_batches = max_epochs * limit_train_batches
    limit_val_batches = 0
    initial_weights = {}

    # train function
    def parity_train_with_one_optimizer(enable_pl_optimizer, vanilla=False, mocked=False):
        seed_everything(42)
        if vanilla:
            # Note: global_step counts training_loop.optimizer_step
            expected_global_step = expected_batches
            if using_amp:
                model = AutomatiocOptimizationVanillaAMPNativeModel("SGD", mocked=mocked)
            else:
                model = AutomatiocOptimizationNativeModel("SGD", mocked=mocked)
            initial_weights["vanilla_model"] = model.layer.weight.clone()
        else:
            expected_global_step = (expected_batches) // accumulate_grad_batches
            if enable_pl_optimizer:
                model = BaseParityAutomaticOptimizationModel("Adam" if mocked else "SGD", mocked=mocked)
                initial_weights["model_wi_pl_optimizer"] = model.layer.weight.clone()
            else:
                model = BaseParityAutomaticOptimizationModel("AdamW" if mocked else "SGD", mocked=mocked)
                initial_weights["model_wo_pl_optimizer"] = model.layer.weight.clone()

        model.training_epoch_end = None

        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            enable_pl_optimizer=enable_pl_optimizer,
            accumulate_grad_batches=accumulate_grad_batches if not vanilla else 1,
            amp_backend=amp_backend,
            precision=precision,
            gpus=1
        )
        trainer.fit(model)

        assert np.abs(trainer.global_step - expected_global_step) <= 2
        return model

    # assertions

    model_wo_pl_optimizer = parity_train_with_one_optimizer(False)
    model_wi_pl_optimizer = parity_train_with_one_optimizer(True)

    assert torch.equal(initial_weights["model_wo_pl_optimizer"], initial_weights["model_wi_pl_optimizer"])
    assert len(model_wo_pl_optimizer.losses) == expected_batches

    assert np.abs(len(model_wo_pl_optimizer.grads) - (expected_batches // accumulate_grad_batches)) <= 2
    assert not torch.FloatTensor(model_wo_pl_optimizer.losses).isnan().any()

    assert torch.equal(torch.FloatTensor(model_wo_pl_optimizer.losses), torch.FloatTensor(model_wi_pl_optimizer.losses))
    assert model_wo_pl_optimizer.on_before_zero_grad_count == model_wi_pl_optimizer.on_before_zero_grad_count

    for b_grad, a_grad in zip(model_wo_pl_optimizer.grads, model_wi_pl_optimizer.grads):
        assert torch.equal(b_grad, a_grad), 'Grad parameters are different'

    for b_w, a_w in zip(model_wo_pl_optimizer.parameters(), model_wi_pl_optimizer.parameters()):
        assert torch.equal(b_w, a_w), 'Model parameters are different'

    vanilla_model = parity_train_with_one_optimizer(False, vanilla=True)

    assert torch.equal(initial_weights["model_wo_pl_optimizer"], initial_weights["vanilla_model"])
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

            parity_train_with_one_optimizer(False, mocked=True)
            parity_train_with_one_optimizer(True, mocked=True)
            parity_train_with_one_optimizer(False, vanilla=True, mocked=True)

        assert mock_sdg_step.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_zero_grad.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_zero_grad.call_count == mock_adam_zero_grad.call_count
        assert mock_sdg_zero_grad.call_count == mock_adamw_zero_grad.call_count

###############################################
# TESTING MANUAL OPTIMIZATION - ONE OPTIMIZER #
###############################################


class BaseParityGANAutomatiocOptimizationModel(BasicGAN):

    def __init__(self, gen_optim="Adam", dis_optim="Adam", mocked=False, lr_gen=0.1, lr_dis=0.1):
        super().__init__()
        self.losses = []
        self.grads = {}
        self.on_before_zero_grad_count = 0
        self.mocked = mocked
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis

    def on_before_zero_grad(self, optimizer):
        self.on_before_zero_grad_count += 1
        if self.layer.weight.grad is not None and not is_overridden("optimizer_step", self):
            self.grads.append(self.layer.weight.grad.clone())

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_g = getattr(torch.optim, self.gen_optim)(self.generator.parameters(), lr=self.lr_gen)
        opt_d = getattr(torch.optim, self.dis_optim)(self.discriminator.parameters(), lr=self.lr_dis)
        return [opt_g, opt_d], []


########################################################
# TESTING AUTOMATIC OPTIMIZATION - MULTIPLE OPTIMIZERS #
########################################################

####################################################
# TESTING MANUAL OPTIMIZATION - MULTIPLE OPTIMIZER #
####################################################
