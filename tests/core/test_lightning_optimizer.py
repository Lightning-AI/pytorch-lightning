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


def test_lightning_optimizer(tmpdir):
    """
    Test that optimizer are correctly wrapped by our LightningOptimizer
    """
    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            # optimizer = LightningOptimizer(self.trainer, optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    groups = "{'dampening': 0, 'initial_lr': 0.1, 'lr': 0.01, 'momentum': 0, 'nesterov': False, 'weight_decay': 0}"
    expected = f"LightningSGD(groups=[{groups}])"
    assert trainer.optimizers[0].__repr__() == expected


def test_lightning_optimizer_from_user(tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended.
    """

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer = LightningOptimizer(optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    groups = "{'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'initial_lr': 0.1, 'lr': 0.01, 'weight_decay': 0}"
    expected = f"LightningAdam(groups=[{groups}])"
    assert trainer.optimizers[0].__repr__() == expected


@patch("torch.optim.Adam.step", autospec=True)
@patch("torch.optim.SGD.step", autospec=True)
def test_lightning_optimizer_manual_optimization(mock_sgd_step, mock_adam_step, tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended for now.
    """
    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            (opt_1, opt_2) = self.optimizers()
            assert isinstance(opt_1, LightningOptimizer)
            assert isinstance(opt_2, LightningOptimizer)

            output = self.layer(batch)
            loss_1 = self.loss(batch, output)
            self.manual_backward(loss_1, opt_1)
            opt_1.step()

            def closure():
                output = self.layer(batch)
                loss_2 = self.loss(batch, output)
                self.manual_backward(loss_2, opt_2)
            opt_2.step(closure=closure)

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer_1 = LightningOptimizer(optimizer_1, 4)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

        @property
        def automatic_optimization(self) -> bool:
            return False

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    assert len(mock_sgd_step.mock_calls) == 2
    assert len(mock_adam_step.mock_calls) == 8


@patch("torch.optim.Adam.step", autospec=True)
@patch("torch.optim.SGD.step", autospec=True)
def test_lightning_optimizer_manual_optimization_and_accumulated_gradients(mock_sgd_step, mock_adam_step, tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended.
    """
    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            (opt_1, opt_2) = self.optimizers()
            assert isinstance(opt_1, LightningOptimizer)
            assert isinstance(opt_2, LightningOptimizer)

            output = self.layer(batch)
            loss_1 = self.loss(batch, output)
            self.manual_backward(loss_1, opt_1)
            opt_1.step()

            def closure():
                output = self.layer(batch)
                loss_2 = self.loss(batch, output)
                self.manual_backward(loss_2, opt_2)
            opt_2.step(closure=closure)

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer_1 = LightningOptimizer(optimizer_1, 4)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

        @property
        def automatic_optimization(self) -> bool:
            return False

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        accumulate_grad_batches=2,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    assert len(mock_sgd_step.mock_calls) == 2
    assert len(mock_adam_step.mock_calls) == 4


def test_state(tmpdir):
    model = torch.nn.Linear(3, 4)
    optimizer = torch.optim.Adam(model.parameters())
    lightning_optimizer = LightningOptimizer(optimizer)

    # test state
    assert optimizer.state == lightning_optimizer.state
    lightning_optimizer.state = optimizer.state
    assert optimizer.state == lightning_optimizer.state

    # test param_groups
    assert optimizer.param_groups == lightning_optimizer.param_groups
    lightning_optimizer.param_groups = optimizer.param_groups
    assert optimizer.param_groups == lightning_optimizer.param_groups

    # test defaults
    assert optimizer.defaults == lightning_optimizer.defaults
    lightning_optimizer.defaults = optimizer.defaults
    assert optimizer.defaults == lightning_optimizer.defaults

    assert isinstance(lightning_optimizer, LightningOptimizer)
    assert isinstance(lightning_optimizer, Adam)
    assert isinstance(lightning_optimizer, Optimizer)
    lightning_dict = {}
    special_attrs = ["_accumulate_grad_batches", "_optimizer", "_optimizer_idx", "_support_closure",
                     "_trainer", "__getstate__", "__setstate__", "state_dict", "load_state_dict",
                     "zero_grad", "__setstate__", "add_param_group"]
    for k, v in lightning_optimizer.__dict__.items():
        if k not in special_attrs:
            lightning_dict[k] = v
    assert lightning_dict == optimizer.__dict__
    assert optimizer.state_dict() == lightning_optimizer.state_dict()
    assert optimizer.state == lightning_optimizer.state


def test_lightning_optimizer_automatic_optimization(tmpdir):
    """
    Test lightning optimize works with make_optimizer_step in automatic_optimization
    """
    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def training_epoch_end(self, outputs):
            outputs = sum(outputs, [])
            torch.stack([x["loss"] for x in outputs]).mean()

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

            assert optimizer_closure.__name__ == "train_step_and_backward_closure"

            optimizer.step(closure=optimizer_closure, make_optimizer_step=batch_idx % 2 == 0)

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer_1 = LightningOptimizer(optimizer_1, 4)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=10,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)


def test_lightning_optimizer_automatic_optimization_optimizer_zero_grad(tmpdir):
    """
    Test lightning optimize works with optimizer_zero_grad overrides in automatic_optimization
    """

    with patch("torch.optim.Adam.zero_grad") as adam_zero_grad, \
         patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:

        class TestModel(BoringModel):

            def training_step(self, batch, batch_idx, optimizer_idx=None):
                output = self.layer(batch)
                loss = self.loss(batch, output)
                return {"loss": loss}

            def training_epoch_end(self, outputs):
                outputs = sum(outputs, [])
                torch.stack([x["loss"] for x in outputs]).mean()

            def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
                if optimizer_idx == 0:
                    if batch_idx % 2 == 0:
                        optimizer.zero_grad()

                if optimizer_idx == 1:
                    if batch_idx % 5 == 0:
                        optimizer.zero_grad()

            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                               optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

                assert optimizer_closure.__name__ == "train_step_and_backward_closure"

                optimizer.step(closure=optimizer_closure)

            def configure_optimizers(self):
                optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
                optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
                return [optimizer_1, optimizer_2], [lr_scheduler]

        model = TestModel()
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            limit_train_batches=10,
            limit_val_batches=1,
            max_epochs=1,
            weights_summary=None,
            enable_pl_optimizer=True,
        )
        trainer.fit(model)

        assert adam_zero_grad.call_count == 2
        assert sgd_zero_grad.call_count == 5


def test_lightning_optimizer_automatic_optimization_optimizer_zero_grad_make_optimizer_step(tmpdir):
    """
    Test lightning optimize works with optimizer_zero_grad overrides and make_optimizer_step in automatic_optimization
    """

    try:
        with patch("torch.optim.Adam.zero_grad") as adam_zero_grad, \
             patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:

            class TestModel(BoringModel):

                def training_step(self, batch, batch_idx, optimizer_idx=None):
                    output = self.layer(batch)
                    loss = self.loss(batch, output)
                    return {"loss": loss}

                def training_epoch_end(self, outputs):
                    outputs = sum(outputs, [])
                    torch.stack([x["loss"] for x in outputs]).mean()

                def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
                    if optimizer_idx == 0:
                        if batch_idx % 2 == 0:
                            optimizer.zero_grad()

                    if optimizer_idx == 1:
                        if batch_idx % 5 == 0:
                            optimizer.zero_grad()

                def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

                    assert optimizer_closure.__name__ == "train_step_and_backward_closure"

                    if optimizer_idx == 0:
                        optimizer.step(closure=optimizer_closure, make_optimizer_step=batch_idx % 3 == 0)
                        return
                    optimizer.step(closure=optimizer_closure)

                def configure_optimizers(self):
                    optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
                    optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
                    return [optimizer_1, optimizer_2], [lr_scheduler]

            model = TestModel()
            trainer = Trainer(
                default_root_dir=os.getcwd(),
                limit_train_batches=20,
                limit_val_batches=1,
                max_epochs=1,
                weights_summary=None,
                enable_pl_optimizer=True,
            )
            trainer.fit(model)

            assert adam_zero_grad.call_count == 4
            assert sgd_zero_grad.call_count == 10

    except MisconfigurationException as e:
        assert "When overriding LightningModule `optimizer_zero_grad`, make_optimizer_step is not allowed" in str(e)


def test_lightning_optimizer_automatic_optimization_make_optimizer_step_2(tmpdir):
    """
    Test lightning optimize works with make_optimizer_step in automatic_optimization
    """

    with patch("torch.optim.Adam.zero_grad") as adam_zero_grad, \
         patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:

        class TestModel(BoringModel):

            def training_step(self, batch, batch_idx, optimizer_idx=None):
                output = self.layer(batch)
                loss = self.loss(batch, output)
                return {"loss": loss}

            def training_epoch_end(self, outputs):
                outputs = sum(outputs, [])
                torch.stack([x["loss"] for x in outputs]).mean()

            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                               optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

                assert optimizer_closure.__name__ == "train_step_and_backward_closure"

                make_optimizer_step = None
                if optimizer_idx == 0:
                    make_optimizer_step = batch_idx % 4 == 0
                optimizer.step(closure=optimizer_closure, make_optimizer_step=make_optimizer_step)

            def configure_optimizers(self):
                optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
                optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
                return [optimizer_1, optimizer_2], [lr_scheduler]

        model = TestModel()
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            weights_summary=None,
            enable_pl_optimizer=True,
        )
        trainer.fit(model)

        assert adam_zero_grad.call_count == 20
        assert sgd_zero_grad.call_count == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(["precision", "amp_backend"], [
    pytest.param(16, "native"),
    pytest.param(32, "native"),
])
@pytest.mark.parametrize('accumulate_grad_batches', [1])
def test_parity_training_lightning_optimizer(tmpdir, amp_backend, precision, accumulate_grad_batches):
    """
    Test training with accumulated gradients with and within enable_pl_optimizer reaches the same weights
    """

    initial_weights = {}

    class TestBoringModel(BoringModel):

        def __init__(self, optimizer_name):
            super().__init__()
            self._optimizer_name = optimizer_name
            self.losses = []
            self.grads = []
            self.on_before_zero_grad_count = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_count += 1
            if self.layer.weight.grad is not None and not is_overridden("optimizer_step", self):
                self.grads.append(self.layer.weight.grad.clone())

        def configure_optimizers(self):
            optimizer = getattr(torch.optim, self._optimizer_name)(self.layer.parameters(), lr=0.1)
            return optimizer

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.losses.append(loss.detach().item())
            return {"loss": loss}

    TestBoringModel.training_epoch_end = None

    max_epochs = np.random.randint(1, 3)
    limit_train_batches = np.random.randint(11, 27)

    print(max_epochs, limit_train_batches)

    expected_batches = max_epochs * limit_train_batches
    is_divisible = limit_train_batches % accumulate_grad_batches == 0
    limit_val_batches = 0

    def train(enable_pl_optimizer, override=False, mock=False):
        seed_everything(42)
        if override:

            def should_accumulate(trainer):
                accumulation_done = (trainer.batch_idx + 1) == trainer.num_training_batches
                is_final_batch = (trainer.batch_idx + 1) % accumulate_grad_batches == 0
                return not (accumulation_done or is_final_batch)

            # Note: global_step counts training_loop.optimizer_step
            expected_global_step = expected_batches

            class TestModel(TestBoringModel):
                grad_checked = False

                def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure,
                                   on_tpu=False, using_native_amp=False, using_lbfgs=False):
                    assert not isinstance(optimizer, LightningOptimizer)

                    closure()
                    if should_accumulate(self.trainer):
                        return

                    self.grad_checked = True
                    assert torch.abs(self.layer.weight.grad).sum() > 0
                    self.grads.append(self.layer.weight.grad.clone())

                    if using_native_amp:
                        self.trainer.scaler.step(optimizer)
                        if mock:
                            optimizer.step()
                        self.trainer.scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    if not mock:
                        assert torch.abs(self.layer.weight.grad).sum() == 0

            model = TestModel("SGD")
            initial_weights["override"] = model.layer.weight.clone()
        else:
            expected_global_step = (expected_batches) // accumulate_grad_batches
            if enable_pl_optimizer:
                model = TestBoringModel("Adam" if mock else "SGD")
                initial_weights["after"] = model.layer.weight.clone()
            else:
                model = TestBoringModel("AdamW" if mock else "SGD")
                initial_weights["before"] = model.layer.weight.clone()

        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            enable_pl_optimizer=enable_pl_optimizer,
            accumulate_grad_batches=accumulate_grad_batches if not override else 1,
            amp_backend=amp_backend,
            precision=precision,
            gpus=1
        )
        trainer.fit(model)
        if is_divisible:
            assert trainer.global_step == expected_global_step
        else:
            assert trainer.global_step == expected_global_step + 1
        return model

    before = train(False)
    after = train(True)

    assert torch.equal(initial_weights["before"], initial_weights["after"])
    assert len(before.losses) == expected_batches

    if is_divisible:
        assert len(before.grads) == (expected_batches // accumulate_grad_batches)
    else:
        assert len(before.grads) == (expected_batches // accumulate_grad_batches) + 1

    assert not torch.FloatTensor(before.losses).isnan().any()

    assert torch.equal(torch.FloatTensor(before.losses), torch.FloatTensor(after.losses))
    assert before.on_before_zero_grad_count == after.on_before_zero_grad_count

    for b_grad, a_grad in zip(before.grads, after.grads):
        assert torch.equal(b_grad, a_grad), 'Grad parameters are different'

    for b_w, a_w in zip(before.parameters(), after.parameters()):
        assert torch.equal(b_w, a_w), 'Model parameters are different'

    override = train(False, override=True)

    assert torch.equal(initial_weights["before"], initial_weights["override"])
    assert override.grad_checked
    assert override.losses == before.losses
    assert (override.on_before_zero_grad_count // accumulate_grad_batches) == before.on_before_zero_grad_count

    for b_grad, o_grad in zip(before.grads, override.grads):
        assert torch.equal(b_grad, o_grad), 'Grad parameters are different'

    for b_w, o_w in zip(before.parameters(), override.parameters()):
        assert torch.equal(b_w, o_w), 'Model parameters are different'

    if precision == 32:

        with patch("torch.optim.SGD.step") as mock_sdg_step, \
             patch("torch.optim.Adam.step") as mock_adam_step, \
             patch("torch.optim.AdamW.step") as mock_adamw_step, \
             patch("torch.optim.SGD.zero_grad") as mock_sdg_zero_grad, \
             patch("torch.optim.Adam.zero_grad") as mock_adam_zero_grad, \
             patch("torch.optim.AdamW.zero_grad") as mock_adamw_zero_grad:

            before = train(False, mock=True)
            after = train(True, mock=True)
            override = train(False, override=True, mock=True)

        assert mock_sdg_step.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_zero_grad.call_count == (expected_batches // accumulate_grad_batches)
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_step.call_count == mock_adam_step.call_count
        assert mock_sdg_zero_grad.call_count == mock_adam_zero_grad.call_count
        assert mock_sdg_zero_grad.call_count == mock_adamw_zero_grad.call_count
