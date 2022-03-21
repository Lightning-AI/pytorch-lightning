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
import pytest
import torch
from torch import optim

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.finetuning import BackboneFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel


def test_lr_monitor_single_lr(tmpdir):
    """Test that learning rates are extracted and logged for single lr scheduler."""
    tutils.reset_seed()

    model = BoringModel()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor]
    )
    trainer.fit(model)

    assert lr_monitor.lrs, "No learning rates logged"
    assert all(v is None for v in lr_monitor.last_momentum_values.values()), "Momentum should not be logged by default"
    assert len(lr_monitor.lrs) == len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ["lr-SGD"]


@pytest.mark.parametrize("opt", ["SGD", "Adam"])
def test_lr_monitor_single_lr_with_momentum(tmpdir, opt: str):
    """Test that learning rates and momentum are extracted and logged for single lr scheduler."""

    class LogMomentumModel(BoringModel):
        def __init__(self, opt):
            super().__init__()
            self.opt = opt

        def configure_optimizers(self):
            if self.opt == "SGD":
                opt_kwargs = {"momentum": 0.9}
            elif self.opt == "Adam":
                opt_kwargs = {"betas": (0.9, 0.999)}

            optimizer = getattr(optim, self.opt)(self.parameters(), lr=1e-2, **opt_kwargs)
            lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=10_000)
            return [optimizer], [lr_scheduler]

    model = LogMomentumModel(opt=opt)
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=2,
        limit_train_batches=5,
        log_every_n_steps=1,
        callbacks=[lr_monitor],
    )
    trainer.fit(model)

    assert all(v is not None for v in lr_monitor.last_momentum_values.values()), "Expected momentum to be logged"
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_scheduler_configs)
    assert all(k == f"lr-{opt}-momentum" for k in lr_monitor.last_momentum_values)


def test_log_momentum_no_momentum_optimizer(tmpdir):
    """Test that if optimizer doesn't have momentum then a warning is raised with log_momentum=True."""

    class LogMomentumModel(BoringModel):
        def configure_optimizers(self):
            optimizer = optim.ASGD(self.parameters(), lr=1e-2)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = LogMomentumModel()
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=2,
        limit_train_batches=5,
        log_every_n_steps=1,
        callbacks=[lr_monitor],
    )
    with pytest.warns(RuntimeWarning, match="optimizers do not have momentum."):
        trainer.fit(model)

    assert all(v == 0 for v in lr_monitor.last_momentum_values.values()), "Expected momentum to be logged"
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_scheduler_configs)
    assert all(k == "lr-ASGD-momentum" for k in lr_monitor.last_momentum_values)


def test_lr_monitor_no_lr_scheduler_single_lr(tmpdir):
    """Test that learning rates are extracted and logged for no lr scheduler."""
    tutils.reset_seed()

    class CustomBoringModel(BoringModel):
        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=0.1)
            return optimizer

    model = CustomBoringModel()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor]
    )

    trainer.fit(model)

    assert lr_monitor.lrs, "No learning rates logged"
    assert len(lr_monitor.lrs) == len(trainer.optimizers)
    assert list(lr_monitor.lrs) == ["lr-SGD"]


@pytest.mark.parametrize("opt", ["SGD", "Adam"])
def test_lr_monitor_no_lr_scheduler_single_lr_with_momentum(tmpdir, opt: str):
    """Test that learning rates and momentum are extracted and logged for no lr scheduler."""

    class LogMomentumModel(BoringModel):
        def __init__(self, opt):
            super().__init__()
            self.opt = opt

        def configure_optimizers(self):
            if self.opt == "SGD":
                opt_kwargs = {"momentum": 0.9}
            elif self.opt == "Adam":
                opt_kwargs = {"betas": (0.9, 0.999)}

            optimizer = getattr(optim, self.opt)(self.parameters(), lr=1e-2, **opt_kwargs)
            return [optimizer]

    model = LogMomentumModel(opt=opt)
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=2,
        limit_train_batches=5,
        log_every_n_steps=1,
        callbacks=[lr_monitor],
    )
    trainer.fit(model)

    assert all(v is not None for v in lr_monitor.last_momentum_values.values()), "Expected momentum to be logged"
    assert len(lr_monitor.last_momentum_values) == len(trainer.optimizers)
    assert all(k == f"lr-{opt}-momentum" for k in lr_monitor.last_momentum_values)


def test_log_momentum_no_momentum_optimizer_no_lr_scheduler(tmpdir):
    """Test that if optimizer doesn't have momentum then a warning is raised with log_momentum=True."""

    class LogMomentumModel(BoringModel):
        def configure_optimizers(self):
            optimizer = optim.ASGD(self.parameters(), lr=1e-2)
            return [optimizer]

    model = LogMomentumModel()
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=2,
        limit_train_batches=5,
        log_every_n_steps=1,
        callbacks=[lr_monitor],
    )
    with pytest.warns(RuntimeWarning, match="optimizers do not have momentum."):
        trainer.fit(model)

    assert all(v == 0 for v in lr_monitor.last_momentum_values.values()), "Expected momentum to be logged"
    assert len(lr_monitor.last_momentum_values) == len(trainer.optimizers)
    assert all(k == "lr-ASGD-momentum" for k in lr_monitor.last_momentum_values)


def test_lr_monitor_no_logger(tmpdir):
    tutils.reset_seed()

    model = BoringModel()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=[lr_monitor], logger=False)

    with pytest.raises(MisconfigurationException, match="`Trainer` that has no logger"):
        trainer.fit(model)


@pytest.mark.parametrize("logging_interval", ["step", "epoch"])
def test_lr_monitor_multi_lrs(tmpdir, logging_interval: str):
    """Test that learning rates are extracted and logged for multi lr schedulers."""
    tutils.reset_seed()

    class CustomBoringModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=1e-2)
            optimizer2 = optim.Adam(self.parameters(), lr=1e-2)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    model = CustomBoringModel()
    model.training_epoch_end = None

    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    log_every_n_steps = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=7,
        limit_val_batches=0.1,
        callbacks=[lr_monitor],
    )
    trainer.fit(model)

    assert lr_monitor.lrs, "No learning rates logged"
    assert len(lr_monitor.lrs) == len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ["lr-Adam", "lr-Adam-1"], "Names of learning rates not set correctly"

    if logging_interval == "step":
        # divide by 2 because we have 2 optimizers
        expected_number_logged = trainer.global_step // 2 // log_every_n_steps
    if logging_interval == "epoch":
        expected_number_logged = trainer.max_epochs

    assert all(len(lr) == expected_number_logged for lr in lr_monitor.lrs.values())


@pytest.mark.parametrize("logging_interval", ["step", "epoch"])
def test_lr_monitor_no_lr_scheduler_multi_lrs(tmpdir, logging_interval: str):
    """Test that learning rates are extracted and logged for multi optimizers but no lr scheduler."""
    tutils.reset_seed()

    class CustomBoringModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=1e-2)
            optimizer2 = optim.Adam(self.parameters(), lr=1e-2)

            return [optimizer1, optimizer2]

    model = CustomBoringModel()
    model.training_epoch_end = None

    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    log_every_n_steps = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=7,
        limit_val_batches=0.1,
        callbacks=[lr_monitor],
    )
    trainer.fit(model)

    assert lr_monitor.lrs, "No learning rates logged"
    assert len(lr_monitor.lrs) == len(trainer.optimizers)
    assert list(lr_monitor.lrs) == ["lr-Adam", "lr-Adam-1"], "Names of learning rates not set correctly"

    if logging_interval == "step":
        # divide by 2 because we have 2 optimizers
        expected_number_logged = trainer.global_step // 2 // log_every_n_steps
    if logging_interval == "epoch":
        expected_number_logged = trainer.max_epochs

    assert all(len(lr) == expected_number_logged for lr in lr_monitor.lrs.values())


def test_lr_monitor_param_groups(tmpdir):
    """Test that learning rates are extracted and logged for single lr scheduler."""
    tutils.reset_seed()

    class CustomClassificationModel(ClassificationModel):
        def configure_optimizers(self):
            param_groups = [
                {"params": list(self.parameters())[:2], "lr": self.lr * 0.1},
                {"params": list(self.parameters())[2:], "lr": self.lr},
            ]

            optimizer = optim.Adam(param_groups)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
            return [optimizer], [lr_scheduler]

    model = CustomClassificationModel()
    dm = ClassifDataModule()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor]
    )
    trainer.fit(model, datamodule=dm)

    assert lr_monitor.lrs, "No learning rates logged"
    assert len(lr_monitor.lrs) == 2 * len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ["lr-Adam/pg1", "lr-Adam/pg2"], "Names of learning rates not set correctly"


def test_lr_monitor_custom_name(tmpdir):
    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer, [scheduler] = super().configure_optimizers()
            lr_scheduler = {"scheduler": scheduler, "name": "my_logging_name"}
            return optimizer, [lr_scheduler]

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_monitor],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    assert list(lr_monitor.lrs) == ["my_logging_name"]


def test_lr_monitor_custom_pg_name(tmpdir):
    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD([{"params": list(self.layer.parameters()), "name": "linear"}], lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=2,
        limit_train_batches=2,
        callbacks=[lr_monitor],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    assert list(lr_monitor.lrs) == ["lr-SGD/linear"]


def test_lr_monitor_duplicate_custom_pg_names(tmpdir):
    tutils.reset_seed()

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.linear_a = torch.nn.Linear(32, 16)
            self.linear_b = torch.nn.Linear(16, 2)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

        def configure_optimizers(self):
            param_groups = [
                {"params": list(self.linear_a.parameters()), "name": "linear"},
                {"params": list(self.linear_b.parameters()), "name": "linear"},
            ]
            optimizer = torch.optim.SGD(param_groups, lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=2,
        limit_train_batches=2,
        callbacks=[lr_monitor],
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.raises(
        MisconfigurationException, match="A single `Optimizer` cannot have multiple parameter groups with identical"
    ):
        trainer.fit(TestModel())


def test_multiple_optimizers_basefinetuning(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.Linear(32, 32), torch.nn.Linear(32, 32), torch.nn.ReLU(True)
            )
            self.layer = torch.nn.Linear(32, 2)

        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def forward(self, x):
            return self.layer(self.backbone(x))

        def configure_optimizers(self):
            parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
            opt = optim.Adam(parameters, lr=0.1)
            opt_2 = optim.Adam(parameters, lr=0.1)
            opt_3 = optim.Adam(parameters, lr=0.1)
            optimizers = [opt, opt_2, opt_3]
            schedulers = [
                optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5),
                optim.lr_scheduler.StepLR(opt_2, step_size=1, gamma=0.5),
            ]
            return optimizers, schedulers

    class Check(Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            num_param_groups = sum(len(opt.param_groups) for opt in trainer.optimizers)

            if trainer.current_epoch == 0:
                assert num_param_groups == 3
            elif trainer.current_epoch == 1:
                assert num_param_groups == 4
                assert list(lr_monitor.lrs) == ["lr-Adam-1", "lr-Adam-2", "lr-Adam/pg1", "lr-Adam/pg2"]
            elif trainer.current_epoch == 2:
                assert num_param_groups == 5
                assert list(lr_monitor.lrs) == [
                    "lr-Adam-2",
                    "lr-Adam/pg1",
                    "lr-Adam/pg2",
                    "lr-Adam-1/pg1",
                    "lr-Adam-1/pg2",
                ]
            else:
                expected = [
                    "lr-Adam-2",
                    "lr-Adam/pg1",
                    "lr-Adam/pg2",
                    "lr-Adam-1/pg1",
                    "lr-Adam-1/pg2",
                    "lr-Adam-1/pg3",
                ]
                assert list(lr_monitor.lrs) == expected

    class TestFinetuning(BackboneFinetuning):
        def freeze_before_training(self, pl_module):
            self.freeze(pl_module.backbone[0])
            self.freeze(pl_module.backbone[1])
            self.freeze(pl_module.layer)

        def finetune_function(self, pl_module, epoch: int, optimizer, opt_idx: int):
            """Called when the epoch begins."""
            if epoch == 1 and opt_idx == 0:
                self.unfreeze_and_add_param_group(pl_module.backbone[0], optimizer, lr=0.1)
            if epoch == 2 and opt_idx == 1:
                self.unfreeze_and_add_param_group(pl_module.layer, optimizer, lr=0.1)

            if epoch == 3 and opt_idx == 1:
                assert len(optimizer.param_groups) == 2
                self.unfreeze_and_add_param_group(pl_module.backbone[1], optimizer, lr=0.1)
                assert len(optimizer.param_groups) == 3

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_val_batches=0,
        limit_train_batches=2,
        callbacks=[TestFinetuning(), lr_monitor, Check()],
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    model = TestModel()
    model.training_epoch_end = None
    trainer.fit(model)

    expected = [0.1, 0.1, 0.1, 0.1, 0.1]
    assert lr_monitor.lrs["lr-Adam-2"] == expected

    expected = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    assert lr_monitor.lrs["lr-Adam/pg1"] == expected

    expected = [0.1, 0.05, 0.025, 0.0125]
    assert lr_monitor.lrs["lr-Adam/pg2"] == expected

    expected = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    assert lr_monitor.lrs["lr-Adam-1/pg1"] == expected

    expected = [0.1, 0.05, 0.025]
    assert lr_monitor.lrs["lr-Adam-1/pg2"] == expected

    expected = [0.1, 0.05]
    assert lr_monitor.lrs["lr-Adam-1/pg3"] == expected


def test_lr_monitor_multiple_param_groups_no_lr_scheduler(tmpdir):
    """Test that the `LearningRateMonitor` is able to log correct keys with multiple param groups and no
    lr_scheduler."""

    class TestModel(BoringModel):
        def __init__(self, lr, momentum):
            super().__init__()
            self.save_hyperparameters()
            self.linear_a = torch.nn.Linear(32, 16)
            self.linear_b = torch.nn.Linear(16, 2)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

        def configure_optimizers(self):
            param_groups = [
                {"params": list(self.linear_a.parameters())},
                {"params": list(self.linear_b.parameters())},
            ]
            optimizer = torch.optim.Adam(param_groups, lr=self.hparams.lr, betas=self.hparams.momentum)
            return optimizer

    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=2,
        limit_train_batches=2,
        callbacks=[lr_monitor],
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    lr = 1e-2
    momentum = 0.7
    model = TestModel(lr=lr, momentum=(momentum, 0.999))
    trainer.fit(model)

    assert len(lr_monitor.lrs) == len(trainer.optimizers[0].param_groups)
    assert list(lr_monitor.lrs) == ["lr-Adam/pg1", "lr-Adam/pg2"]
    assert list(lr_monitor.last_momentum_values) == ["lr-Adam/pg1-momentum", "lr-Adam/pg2-momentum"]
    assert all(val == momentum for val in lr_monitor.last_momentum_values.values())
    assert all(all(val == lr for val in lr_monitor.lrs[lr_key]) for lr_key in lr_monitor.lrs)
