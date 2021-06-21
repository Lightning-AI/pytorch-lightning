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
import gc
from typing import Any
from unittest.mock import DEFAULT, patch

import torch
from torch.optim import Adam, Optimizer, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.helpers.boring_model import BoringModel


def test_lightning_optimizer(tmpdir):
    """
    Test that optimizer are correctly wrapped by our LightningOptimizer
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    groups = "{'dampening': 0, 'initial_lr': 0.1, 'lr': 0.01, 'momentum': 0, 'nesterov': False, 'weight_decay': 0}"
    expected = f"LightningSGD(groups=[{groups}])"
    assert trainer._lightning_optimizers[0].__repr__() == expected


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
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    groups = "{'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'initial_lr': 0.1, 'lr': 0.01, 'weight_decay': 0}"
    expected = f"LightningAdam(groups=[{groups}])"
    assert trainer._lightning_optimizers[0].__repr__() == expected


def test_lightning_optimizer_manual_optimization_and_accumulated_gradients(tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended.
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
                output = self.layer(batch)
                loss = self.loss(batch, output)
                opt.zero_grad()
                self.manual_backward(loss)

            if batch_idx % 2 == 0:
                closure(opt_1)
                opt_1.step()

            closure(opt_2)
            opt_2.step()

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )

    with patch.multiple(torch.optim.SGD, zero_grad=DEFAULT, step=DEFAULT) as sgd, \
         patch.multiple(torch.optim.Adam, zero_grad=DEFAULT, step=DEFAULT) as adam:
        trainer.fit(model)

    assert sgd["step"].call_count == 4
    assert adam["step"].call_count == 8

    assert sgd["zero_grad"].call_count == 4
    assert adam["zero_grad"].call_count == 8


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
    special_attrs = [
        "_accumulate_grad_batches",
        "_optimizer",
        "_optimizer_idx",
        "_support_closure",
        "_trainer",
        "__getstate__",
        "__setstate__",
        "state_dict",
        "load_state_dict",
        "zero_grad",
        "__setstate__",
        "add_param_group",
        "_total_optimizer_step_calls",
    ]

    for k, v in lightning_optimizer.__dict__.items():
        if k not in special_attrs:
            lightning_dict[k] = v

    assert lightning_dict == optimizer.__dict__
    assert optimizer.state_dict() == lightning_optimizer.state_dict()
    assert optimizer.state == lightning_optimizer.state


def test_lightning_optimizer_automatic_optimization_optimizer_zero_grad(tmpdir):
    """
    Test overriding zero_grad works in automatic_optimization
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs):
            ...

        def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
            if isinstance(optimizer, SGD) and batch_idx % 2 == 0:
                optimizer.zero_grad()
            if isinstance(optimizer, Adam) and batch_idx % 5 == 0:
                optimizer.zero_grad()

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=20,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )

    with patch("torch.optim.Adam.zero_grad") as adam_zero_grad, \
         patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        trainer.fit(model)

    assert adam_zero_grad.call_count == 4
    assert sgd_zero_grad.call_count == 10


def test_lightning_optimizer_automatic_optimization_optimizer_step(tmpdir):
    """
    Test overriding step works in automatic_optimization
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs):
            ...

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **_):
            assert optimizer_closure.__name__ == "training_step_and_backward_closure"
            # not passing the closure to the optimizer because step is mocked
            # zero_grad is called inside the closure
            if isinstance(optimizer, SGD) and batch_idx % 2 == 0:
                optimizer_closure()
                optimizer.step()
            if isinstance(optimizer, Adam) and batch_idx % 4 == 0:
                optimizer_closure()
                optimizer.step()  # not passing the closure here because it's a mock

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )

    with patch.multiple(torch.optim.SGD, zero_grad=DEFAULT, step=DEFAULT) as sgd, \
         patch.multiple(torch.optim.Adam, zero_grad=DEFAULT, step=DEFAULT) as adam:
        trainer.fit(model)

    assert sgd["step"].call_count == 4
    assert adam["step"].call_count == 2

    assert sgd["zero_grad"].call_count == 4
    assert adam["zero_grad"].call_count == 2


def test_lightning_optimizer_automatic_optimization_lbfgs_zero_grad(tmpdir):
    """
    Test zero_grad is called the same number of times as LBFGS requires
    for reevaluation of the loss in automatic_optimization.
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            return torch.optim.LBFGS(self.parameters())

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
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
            if mod_class != 'Linear':
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
            d = {'params': params, 'mod': mod, 'layer_type': mod_class}
            self.params.append(d)

        super(OptimizerWithHooks, self).__init__(self.params, {"lr": 0.01})

    def _save_input(self, mod, i):
        """Saves input of layer"""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, _, grad_output):
        """
        Saves grad on output of layer to
        grad is scaled with batch_size since gradient is spread over samples in mini batch
        """
        batch_size = grad_output[0].shape[0]
        if mod.training:
            self.state[mod]['grad'] = grad_output[0] * batch_size

    def step(self, closure=None):
        closure()
        for group in self.param_groups:
            _ = self.state[group['mod']]['x']
            _ = self.state[group['mod']]['grad']
        return True


def test_lightning_optimizer_keeps_hooks(tmpdir):

    class TestModel(BoringModel):
        count_on_train_batch_start = 0
        count_on_train_batch_end = 0

        def configure_optimizers(self):
            return OptimizerWithHooks(self)

        def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
            self.count_on_train_batch_start += 1
            optimizer = self.optimizers(use_pl_optimizer=False)
            assert len(optimizer._fwd_handles) == 1

        def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
            self.count_on_train_batch_end += 1
            del self.trainer._lightning_optimizers
            gc.collect()  # not necessary, just in case

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=4, limit_val_batches=1, max_epochs=1)
    model = TestModel()
    trainer.fit(model)
    assert model.count_on_train_batch_start == 4
    assert model.count_on_train_batch_end == 4
