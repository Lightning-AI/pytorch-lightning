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
from copy import deepcopy
from unittest.mock import DEFAULT, Mock, patch

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.automatic import Closure
from lightning.pytorch.tuner.tuning import Tuner
from torch.optim import SGD, Adam, Optimizer

from tests_pytorch.helpers.runif import RunIf


@pytest.mark.parametrize("auto", [True, False])
def test_lightning_optimizer(tmp_path, auto):
    """Test that optimizer are correctly wrapped by our LightningOptimizer."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            if not auto:
                # note: this is not recommended, only done for coverage
                optimizer = LightningOptimizer(optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path, limit_train_batches=1, limit_val_batches=1, max_epochs=1, enable_model_summary=False
    )
    trainer.fit(model)

    lightning_opt = model.optimizers()
    assert str(lightning_opt) == "Lightning" + str(lightning_opt.optimizer)


def test_init_optimizers_resets_lightning_optimizers(tmp_path):
    """Test that the Trainer resets the `lightning_optimizers` list everytime new optimizers get initialized."""

    def compare_optimizers():
        assert trainer.strategy._lightning_optimizers[0].optimizer is trainer.optimizers[0]

    model = BoringModel()
    model.lr = 0.2
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    tuner.lr_find(model)
    compare_optimizers()

    trainer.fit(model)
    compare_optimizers()

    trainer.fit_loop.max_epochs = 2  # simulate multiple fit calls
    trainer.fit(model)
    compare_optimizers()


def test_lightning_optimizer_manual_optimization_and_accumulated_gradients(tmp_path):
    """Test that the user can use our LightningOptimizer.

    Not recommended.

    """

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            opt_1, opt_2 = self.optimizers()

            assert isinstance(opt_1, LightningOptimizer)
            assert isinstance(opt_2, LightningOptimizer)

            def closure(opt):
                loss = self.step(batch)
                opt.zero_grad()
                self.manual_backward(loss)

            if batch_idx % 2 == 0:
                closure(opt_1)
                opt_1.step()

            closure(opt_2)
            step_output = opt_2.step()
            # check that the step output is returned with manual optimization
            # since the optimizer is mocked, the step output is a Mock
            assert isinstance(step_output, Mock)

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path, limit_train_batches=8, limit_val_batches=1, max_epochs=1, enable_model_summary=False
    )

    with patch.multiple(torch.optim.SGD, zero_grad=DEFAULT, step=DEFAULT) as sgd, patch.multiple(
        torch.optim.Adam, zero_grad=DEFAULT, step=DEFAULT
    ) as adam:
        trainer.fit(model)

    assert sgd["step"].call_count == 4
    assert adam["step"].call_count == 8

    assert sgd["zero_grad"].call_count == 4
    assert adam["zero_grad"].call_count == 8


def test_state():
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

    assert optimizer.state_dict() == lightning_optimizer.state_dict()


def test_state_mutation():
    model = torch.nn.Linear(3, 4)
    optimizer0 = torch.optim.Adam(model.parameters(), lr=0.1)
    lightning_optimizer0 = LightningOptimizer(optimizer0)

    optimizer0.param_groups[0]["lr"] = 1.0
    assert lightning_optimizer0.param_groups[0]["lr"] == 1.0

    # Load state into the unwrapped optimizer
    state_dict0 = deepcopy(optimizer0.state_dict())
    optimizer1 = torch.optim.Adam(model.parameters(), lr=100)
    lightning_optimizer1 = LightningOptimizer(optimizer1)
    optimizer1.load_state_dict(state_dict0)
    assert lightning_optimizer1.param_groups[0]["lr"] == 1.0

    # Load state into wrapped optimizer
    optimizer2 = torch.optim.Adam(model.parameters(), lr=100)
    lightning_optimizer2 = LightningOptimizer(optimizer2)
    lightning_optimizer2.load_state_dict(state_dict0)
    assert lightning_optimizer2.param_groups[0]["lr"] == 1.0


def test_lightning_optimizer_automatic_optimization_optimizer_zero_grad(tmp_path):
    """Test overriding zero_grad works in automatic_optimization."""

    class TestModel(BoringModel):
        def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
            if batch_idx % 2 == 0:
                optimizer.zero_grad()

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path, limit_train_batches=20, limit_val_batches=1, max_epochs=1, enable_model_summary=False
    )

    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        trainer.fit(model)
    assert sgd_zero_grad.call_count == 10


def test_lightning_optimizer_automatic_optimization_optimizer_step(tmp_path):
    """Test overriding step works in automatic_optimization."""

    class TestModel(BoringModel):
        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **_):
            assert isinstance(optimizer_closure, Closure)
            # zero_grad is called inside the closure
            optimizer_closure()
            # not passing the closure to the optimizer because step is mocked
            if batch_idx % 2 == 0:
                optimizer.step()

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1], [lr_scheduler]

    model = TestModel()

    limit_train_batches = 8
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        max_epochs=1,
        enable_model_summary=False,
    )

    with patch.multiple(torch.optim.SGD, zero_grad=DEFAULT, step=DEFAULT) as sgd:
        trainer.fit(model)

    assert sgd["step"].call_count == limit_train_batches // 2
    assert sgd["zero_grad"].call_count == limit_train_batches


@RunIf(mps=False)  # mps does not support LBFGS
def test_lightning_optimizer_automatic_optimization_lbfgs_zero_grad(tmp_path):
    """Test zero_grad is called the same number of times as LBFGS requires for reevaluation of the loss in
    automatic_optimization."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.LBFGS(self.parameters())

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path, limit_train_batches=1, limit_val_batches=1, max_epochs=1, enable_model_summary=False
    )

    with patch("torch.optim.LBFGS.zero_grad") as zero_grad:
        trainer.fit(model)

    lbfgs = model.optimizers()
    max_iter = lbfgs.param_groups[0]["max_iter"]
    assert zero_grad.call_count == max_iter


class OptimizerWithHooks(Optimizer):
    def __init__(self, model):
        self._fwd_handles = []
        self._bwd_handles = []
        self.params = []
        for _, mod in model.named_modules():
            mod_class = mod.__class__.__name__
            if mod_class != "Linear":
                continue

            handle = mod.register_forward_pre_hook(self._save_input)  # save the inputs
            self._fwd_handles.append(handle)  # collect forward-save-input hooks in list
            handle = mod.register_backward_hook(self._save_grad_output)  # save the gradients
            self._bwd_handles.append(handle)  # collect backward-save-grad hook in list

            # save the parameters
            params = [mod.weight]
            if mod.bias is not None:
                params.append(mod.bias)

            # save a param_group for each module
            d = {"params": params, "mod": mod, "layer_type": mod_class}
            self.params.append(d)

        super().__init__(self.params, {"lr": 0.01})

    def _save_input(self, mod, i):
        """Saves input of layer."""
        if mod.training:
            self.state[mod]["x"] = i[0]

    def _save_grad_output(self, mod, _, grad_output):
        """Saves grad on output of layer to grad is scaled with batch_size since gradient is spread over samples in
        mini batch."""
        batch_size = grad_output[0].shape[0]
        if mod.training:
            self.state[mod]["grad"] = grad_output[0] * batch_size

    def step(self, closure=None):
        closure()
        for group in self.param_groups:
            _ = self.state[group["mod"]]["x"]
            _ = self.state[group["mod"]]["grad"]
        return True


def test_lightning_optimizer_keeps_hooks():
    model = BoringModel()
    optimizer = OptimizerWithHooks(model)
    lightning_optimizer = LightningOptimizer(optimizer)
    assert len(optimizer._fwd_handles) == 1
    del lightning_optimizer
    assert len(optimizer._fwd_handles) == 1


def test_params_groups_and_state_are_accessible(tmp_path):
    class TestModel(BoringModel):
        def on_train_start(self):
            # Update the learning rate manually on the unwrapped optimizer
            assert not isinstance(self.trainer.optimizers[0], LightningOptimizer)
            self.trainer.optimizers[0].param_groups[0]["lr"] = 2.0

        def training_step(self, batch, batch_idx):
            opt = self.optimizers()
            assert opt.param_groups[0]["lr"] == 2.0

            loss = self.step(batch)
            self.__loss = loss
            return loss

        def configure_optimizers(self):
            return SGD(self.layer.parameters(), lr=0.1)

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **__):
            # check attributes are accessible
            assert all("lr" in pg for pg in optimizer.param_groups)
            assert optimizer.state is optimizer._optimizer.state
            assert optimizer.defaults is optimizer._optimizer.defaults

            loss = optimizer.step(closure=optimizer_closure)
            # the optimizer step still returns the loss
            assert loss == self.__loss

    model = TestModel()
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, limit_train_batches=1, limit_val_batches=0)
    trainer.fit(model)
