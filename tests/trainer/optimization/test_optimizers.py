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
from unittest import mock
from unittest.mock import call, patch

import pytest
import torch
from torch import optim

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import (
    _configure_optimizers,
    _configure_schedulers_automatic_opt,
    _init_optimizers_and_lr_schedulers,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import LRSchedulerConfig
from tests.helpers.boring_model import BoringDataModule, BoringModel
from tests.helpers.runif import RunIf


def test_optimizer_with_scheduling(tmpdir):
    """Verify that learning rate scheduling is working."""

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2, val_check_interval=0.5
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    init_lr = 0.1
    adjusted_lr = [pg["lr"] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_scheduler_configs) == 1
    assert all(a == adjusted_lr[0] for a in adjusted_lr)
    assert init_lr * 0.1 == adjusted_lr[0]


def test_multi_optimizer_with_scheduling(tmpdir):
    """Verify that learning rate scheduling is working."""

    class TestModel(BoringModel):
        init_lr = 5e-4

        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=self.init_lr)
            optimizer2 = optim.Adam(self.parameters(), lr=self.init_lr)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    adjusted_lr1 = [pg["lr"] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg["lr"] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_scheduler_configs) == 2
    assert all(a == adjusted_lr1[0] for a in adjusted_lr1)
    assert all(a == adjusted_lr2[0] for a in adjusted_lr2)
    assert model.init_lr * 0.1 == adjusted_lr1[0]
    assert model.init_lr * 0.1 == adjusted_lr2[0]


def test_reducelronplateau_with_no_monitor_raises(tmpdir):
    """Test exception when a ReduceLROnPlateau is used with no monitor."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: ([optimizer], [optim.lr_scheduler.ReduceLROnPlateau(optimizer)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(
        MisconfigurationException, match="`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
    ):
        trainer.fit(model)


def test_reducelronplateau_with_no_monitor_in_lr_scheduler_dict_raises(tmpdir):
    """Test exception when lr_scheduler dict has a ReduceLROnPlateau with no monitor."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer)},
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="must include a monitor when a `ReduceLROnPlateau`"):
        trainer.fit(model)


def test_onecyclelr_with_epoch_interval_warns():
    """Test warning when a OneCycleLR is used and interval is epoch."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = {"scheduler": optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=3)}
    with pytest.warns(RuntimeWarning, match="Are you sure you didn't mean 'interval': 'step'?"):
        _configure_schedulers_automatic_opt([lr_scheduler], None)


def test_reducelronplateau_scheduling(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters())
            return {
                "optimizer": optimizer,
                "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "foo",
            }

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    lr_scheduler = trainer.lr_scheduler_configs[0]
    assert lr_scheduler == LRSchedulerConfig(
        scheduler=lr_scheduler.scheduler,
        monitor="foo",
        interval="epoch",
        frequency=1,
        reduce_on_plateau=True,
        strict=True,
        opt_idx=0,
        name=None,
    )


def test_optimizer_return_options(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir)
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer

    # single optimizer
    opt_a = optim.Adam(model.parameters(), lr=0.002)
    opt_b = optim.SGD(model.parameters(), lr=0.002)
    scheduler_a = optim.lr_scheduler.StepLR(opt_a, 10)
    scheduler_b = optim.lr_scheduler.StepLR(opt_b, 10)

    # single optimizer
    model.configure_optimizers = lambda: opt_a
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == 1 and len(lr_sched) == len(freq) == 0

    # opt tuple
    model.configure_optimizers = lambda: (opt_a, opt_b)
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    # opt list
    model.configure_optimizers = lambda: [opt_a, opt_b]
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    ref_lr_sched = LRSchedulerConfig(
        scheduler=scheduler_a,
        interval="epoch",
        frequency=1,
        reduce_on_plateau=False,
        monitor=None,
        strict=True,
        name=None,
        opt_idx=0,
    )

    # opt tuple of 2 lists
    model.configure_optimizers = lambda: ([opt_a], [scheduler_a])
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt tuple of 1 list
    model.configure_optimizers = lambda: ([opt_a], scheduler_a)
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt single dictionary
    model.configure_optimizers = lambda: {"optimizer": opt_a, "lr_scheduler": scheduler_a}
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt multiple dictionaries with frequencies
    model.configure_optimizers = lambda: (
        {"optimizer": opt_a, "lr_scheduler": scheduler_a, "frequency": 1},
        {"optimizer": opt_b, "lr_scheduler": scheduler_b, "frequency": 5},
    )
    opt, lr_sched, freq = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == len(freq) == 2
    assert opt[0] == opt_a
    ref_lr_sched.opt_idx = 0
    assert lr_sched[0] == ref_lr_sched
    ref_lr_sched.scheduler = scheduler_b
    ref_lr_sched.opt_idx = 1
    assert lr_sched[1] == ref_lr_sched
    assert freq == [1, 5]


def test_none_optimizer(tmpdir):
    model = BoringModel()
    model.configure_optimizers = lambda: None
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)
    with pytest.warns(UserWarning, match="will run with no optimizer"):
        trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_configure_optimizer_from_dict(tmpdir):
    """Tests if `configure_optimizer` method could return a dictionary with `optimizer` field only."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            config = {"optimizer": optim.SGD(params=self.parameters(), lr=1e-03)}
            return config

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    "schedulers, kwargs, intervals, frequencies, expected_steps, max_epochs",
    [
        (
            (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.OneCycleLR),
            (dict(max_lr=0.01, total_steps=3), dict(max_lr=0.01, total_steps=2)),
            ("step", "step"),
            (3, 2),
            (4, 3),
            1,
        ),
        (
            (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.OneCycleLR),
            (dict(max_lr=0.01, total_steps=5), dict(max_lr=0.01, total_steps=5)),
            ("step", "step"),
            (None, None),
            (6, 6),
            1,
        ),
        (
            (optim.lr_scheduler.StepLR, optim.lr_scheduler.CosineAnnealingLR),
            (dict(step_size=5), dict(T_max=2)),
            ("epoch", "epoch"),
            (5, 10),
            (2, 3),
            3,
        ),
    ],
)
def test_step_scheduling_for_multiple_optimizers_with_frequency(
    tmpdir, schedulers, kwargs, intervals, frequencies, expected_steps, max_epochs
):
    """Test that step LR schedulers for multiple optimizers follow the optimizer frequencies when corresponding
    frequency is set."""

    class DummyModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs) -> None:
            pass

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.01)
            optimizer2 = optim.Adam(self.parameters(), lr=0.01)

            lr_scheduler_config_1 = {"scheduler": schedulers[0](optimizer1, **kwargs[0]), "interval": intervals[0]}
            lr_scheduler_config_2 = {"scheduler": schedulers[1](optimizer2, **kwargs[1]), "interval": intervals[1]}

            return [
                {"optimizer": optimizer1, "frequency": frequencies[0], "lr_scheduler": lr_scheduler_config_1},
                {"optimizer": optimizer2, "frequency": frequencies[1], "lr_scheduler": lr_scheduler_config_2},
            ]

    model = DummyModel()

    trainer = Trainer(default_root_dir=tmpdir, limit_val_batches=1, limit_train_batches=5, max_epochs=max_epochs)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert trainer.lr_scheduler_configs[0].opt_idx == 0
    assert trainer.lr_scheduler_configs[1].opt_idx == 1
    # Step count is 1 greater than the expected value because scheduler.step() is called once during initialization
    assert trainer.lr_scheduler_configs[0].scheduler._step_count == expected_steps[0]
    assert trainer.lr_scheduler_configs[1].scheduler._step_count == expected_steps[1]


@pytest.mark.parametrize("fn", ("validate", "test", "predict"))
def test_init_optimizers_during_evaluation_and_prediction(tmpdir, fn):
    """Test that optimizers is an empty list during evaluation and prediction."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.1)
            optimizer2 = optim.Adam(self.parameters(), lr=0.1)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)
    train_fn = getattr(trainer, fn)
    train_fn(TestModel(), datamodule=BoringDataModule(), ckpt_path=None)

    assert len(trainer.lr_scheduler_configs) == 0
    assert len(trainer.optimizers) == 0
    assert len(trainer.optimizer_frequencies) == 0


def test_multiple_optimizers_callbacks(tmpdir):
    """Tests that multiple optimizers can be used with callbacks."""

    class CB(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            pass

        def on_train_epoch_start(self, trainer, pl_module):
            pass

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer_1 = torch.nn.Linear(32, 2)
            self.layer_2 = torch.nn.Linear(32, 2)

        def training_step(self, batch, batch_idx, optimizer_idx):
            if optimizer_idx == 0:
                a = batch[0]
                acc = self.layer_1(a)
            else:
                a = batch[0]
                acc = self.layer_2(a)

            acc = self.loss(acc, acc)
            return acc

        def configure_optimizers(self):
            a = optim.RMSprop(self.layer_1.parameters(), 1e-2)
            b = optim.RMSprop(self.layer_2.parameters(), 1e-2)
            return a, b

    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        callbacks=[CB()],
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=2,
        max_epochs=1,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize("complete_epoch", [True, False])
@mock.patch("torch.optim.lr_scheduler.ReduceLROnPlateau.step")
def test_lr_scheduler_strict(step_mock, tmpdir, complete_epoch):
    """Test "strict" support in lr_scheduler dict."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    max_epochs = 1 if complete_epoch else None
    max_steps = -1 if complete_epoch else 1
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, max_steps=max_steps)

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "giraffe", "strict": True},
    }

    if complete_epoch:
        with pytest.raises(
            MisconfigurationException,
            match=r"ReduceLROnPlateau conditioned on metric .* which is not available\. Available metrics are:",
        ):
            trainer.fit(model)
    else:
        trainer.fit(model)

    step_mock.assert_not_called()

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "giraffe", "strict": False},
    }

    if complete_epoch:
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, max_steps=max_steps)
        with pytest.warns(
            RuntimeWarning, match=r"ReduceLROnPlateau conditioned on metric .* which is not available but strict"
        ):
            trainer.fit(model)

    step_mock.assert_not_called()


def test_unknown_configure_optimizers_raises(tmpdir):
    """Test exception with an unsupported configure_optimizers return."""
    model = BoringModel()
    model.configure_optimizers = lambda: 1
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


def test_optimizer_config_dict_with_extra_keys_warns(tmpdir):
    """Test exception when optimizer configuration dict has extra keys."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    optim_conf = {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1)},
        "foo": 1,
        "bar": 2,
    }
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the optimizer configuration: \{.+\}"):
        _configure_optimizers(optim_conf)


def test_multiple_optimizer_config_dicts_with_extra_keys_warns(tmpdir):
    """Test exception when multiple optimizer configuration dicts have extra keys."""
    model = BoringModel()
    optimizer1 = optim.Adam(model.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler_config_1 = {"scheduler": optim.lr_scheduler.StepLR(optimizer1, 1)}
    lr_scheduler_config_2 = {"scheduler": optim.lr_scheduler.StepLR(optimizer2, 1)}
    optim_conf = [
        {"optimizer": optimizer1, "lr_scheduler": lr_scheduler_config_1, "foo": 1, "bar": 2},
        {"optimizer": optimizer2, "lr_scheduler": lr_scheduler_config_2, "foo": 1, "bar": 2},
    ]
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the optimizer configuration: \{.+\}"):
        _configure_optimizers(optim_conf)


def test_lr_scheduler_with_unknown_interval_raises(tmpdir):
    """Test exception when lr_scheduler dict has unknown interval param value."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1), "interval": "incorrect_unknown_value"},
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match=r'The "interval" key in lr scheduler dict must be'):
        trainer.fit(model)


def test_lr_scheduler_with_extra_keys_warns(tmpdir):
    """Test warning when lr_scheduler dict has extra keys."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1), "foo": 1, "bar": 2},
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the lr scheduler dict: \{.+\}"):
        trainer.fit(model)


def test_lr_scheduler_with_no_actual_scheduler_raises(tmpdir):
    """Test exception when lr_scheduler dict has no scheduler."""
    model = BoringModel()
    model.configure_optimizers = lambda: {"optimizer": optim.Adam(model.parameters()), "lr_scheduler": {}}
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='The lr scheduler dict must have the key "scheduler"'):
        trainer.fit(model)


def test_invalid_optimizer_in_scheduler(tmpdir):
    """Test exception when optimizer attached to lr_schedulers wasn't returned."""

    class InvalidOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt1 = optim.SGD(self.layer.parameters(), lr=0.1)
            opt2 = optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = optim.lr_scheduler.StepLR(opt2, step_size=1)
            return [opt1], [lr_scheduler]

    model = InvalidOptimizerModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="attached with an optimizer that wasn't returned"):
        trainer.fit(model)


def test_invalid_opt_idx_in_scheduler(tmpdir):
    """Test exception when incorrect opt_idx is set in lr_scheduler config."""

    class InvalidOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt1 = optim.SGD(self.layer.parameters(), lr=0.1)
            opt2 = optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = {"scheduler": optim.lr_scheduler.StepLR(opt2, step_size=1), "opt_idx": 0}
            return [opt1, opt2], [lr_scheduler]

    model = InvalidOptimizerModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(
        MisconfigurationException, match="`opt_idx` .* does not match with the index of the respective optimizer"
    ):
        trainer.fit(model)


def test_invalid_optimizer_dict_raises(tmpdir):
    """Test exception when lr_scheduler dict has no scheduler."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return [{"optimizer": optim.Adam(self.parameters())}, optim.Adam(self.parameters())]

    model = DummyModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


def test_warn_invalid_scheduler_key_in_manual_optimization(tmpdir):
    """Test warning when invalid scheduler keys are provided in manual optimization."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def configure_optimizers(self):
            opt = optim.SGD(self.layer.parameters(), lr=0.1)
            sch = optim.lr_scheduler.StepLR(opt, step_size=1)
            return [opt], [{"scheduler": sch, "interval": "epoch"}]

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.warns(RuntimeWarning, match="the keys will be ignored"):
        trainer.fit(model)


@RunIf(min_gpus=2, standalone=True)
def test_optimizer_state_on_device(tmpdir):
    """Test that optimizers that create state initially at instantiation still end up with the state on the GPU."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            # Adagrad creates state tensors immediately, model is not yet on GPU.
            return optim.Adagrad(self.parameters())

        def on_train_start(self, *args, **kwargs):
            opt = self.optimizers()
            _, state = next(iter(opt.state.items()))
            assert state["sum"].device == torch.device("cuda", self.local_rank) == self.device

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, gpus=2, strategy="ddp", fast_dev_run=True)
    trainer.fit(model)


@pytest.mark.parametrize("check_val_every_n_epoch", [1, 2])
@mock.patch("torch.optim.lr_scheduler.StepLR.step")
def test_lr_scheduler_epoch_step_frequency(mocked_sched, check_val_every_n_epoch, tmpdir):
    epochs = 4
    expected_steps = epochs + 1  # every LRScheduler gets called once at init

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        check_val_every_n_epoch=check_val_every_n_epoch,
        max_epochs=epochs,
    )
    trainer.fit(model)
    assert mocked_sched.call_count == expected_steps


@pytest.mark.parametrize("every_n_train_steps, epoch_interval", [(None, True), (2, False), (2, True)])
def test_lr_scheduler_state_updated_before_saving(tmpdir, every_n_train_steps, epoch_interval):
    batches = 2
    max_epochs = 1
    lr, gamma = 1, 10
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        logger=False,
        max_epochs=max_epochs,
        limit_train_batches=batches,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=every_n_train_steps)],
    )

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
            lr_scheduler_config = {"scheduler": lr_scheduler}
            if not epoch_interval:
                lr_scheduler_config["interval"] = "step"
            return [optimizer], [lr_scheduler_config]

        def on_save_checkpoint(self, checkpoint):
            lr_scheduler_config = checkpoint["lr_schedulers"][0]
            # 2 batches ran. since the lr_scheduler_config interval is `step`, the step count should be 2
            assert self.trainer.global_step == batches
            compare_to = max_epochs if epoch_interval else batches
            assert lr_scheduler_config["_step_count"] - 1 == compare_to  # step count starts at 1
            assert lr_scheduler_config["_last_lr"] == [lr * gamma ** compare_to]
            self.on_save_checkpoint_called = True

    model = TestModel()
    trainer.fit(model)
    assert model.on_save_checkpoint_called


@pytest.mark.parametrize("save_on_train_epoch_end", (False, True))
def test_plateau_scheduler_lr_step_interval_updated_after_saving(tmpdir, save_on_train_epoch_end):
    batches = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        logger=False,
        max_epochs=1,
        limit_train_batches=batches,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_on_train_epoch_end=save_on_train_epoch_end)],
    )

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer_1 = torch.optim.Adam(self.parameters())
            optimizer_2 = torch.optim.Adam(self.parameters())

            lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1)
            lr_scheduler_config_1 = {"scheduler": lr_scheduler1, "interval": "step", "monitor": "foo"}

            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=1)
            lr_scheduler_config_2 = {"scheduler": lr_scheduler2, "interval": "step"}
            return [optimizer_1, optimizer_2], [lr_scheduler_config_1, lr_scheduler_config_2]

        def on_save_checkpoint(self, checkpoint):
            lr_scheduler_config_1 = checkpoint["lr_schedulers"][0]
            last_epoch = lr_scheduler_config_1["last_epoch"]
            assert last_epoch == batches - (not save_on_train_epoch_end)  # last epoch starts at 0

            lr_scheduler_config_2 = checkpoint["lr_schedulers"][1]
            assert lr_scheduler_config_2["_step_count"] - 1 == batches  # step count starts at 1

            self.on_save_checkpoint_called = True

    model = TestModel()
    model.training_epoch_end = None
    trainer.fit(model)
    assert model.on_save_checkpoint_called


def test_lr_scheduler_step_hook(tmpdir):
    """Test that custom lr scheduler works and `lr_scheduler_step` is called at appropriate time."""

    class CustomEpochScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self, epoch):
            ...

        def state_dict(self):
            ...

        def load_state_dict(self, state_dict):
            ...

    class CustomBoringModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx=0):
            return super().training_step(batch, batch_idx)

        def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
            # step-level
            if optimizer_idx == 0:
                super().lr_scheduler_step(scheduler, optimizer_idx, metric)
            # epoch-level
            elif optimizer_idx == 1:
                scheduler.step(epoch=self.current_epoch)

        def configure_optimizers(self):
            opt1 = torch.optim.SGD(self.layer.parameters(), lr=1e-2)
            lr_scheduler1 = {"scheduler": torch.optim.lr_scheduler.StepLR(opt1, step_size=1), "interval": "step"}
            opt2 = torch.optim.SGD(self.layer.parameters(), lr=1e-2)
            lr_scheduler2 = CustomEpochScheduler(opt2)
            return {"optimizer": opt1, "lr_scheduler": lr_scheduler1}, {
                "optimizer": opt2,
                "lr_scheduler": lr_scheduler2,
            }

    model = CustomBoringModel()
    model.training_epoch_end = None
    max_epochs = 3
    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_checkpointing=False,
        logger=False,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
    )

    with patch.object(CustomEpochScheduler, "step") as mock_method_epoch, patch.object(
        torch.optim.lr_scheduler.StepLR, "step"
    ) as mock_method_step:
        trainer.fit(model)

    assert mock_method_epoch.mock_calls == [call(epoch=e) for e in range(max_epochs)]
    # first step is called by PyTorch _LRScheduler
    assert mock_method_step.call_count == max_epochs * limit_train_batches + 1


def test_invalid_scheduler_missing_state_dict():
    """Test that custom lr scheduler raises an error if it's missing the state dict."""

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self):
            ...

    class CustomBoringModel(BoringModel):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=1e-2)
            lr_scheduler = CustomScheduler(opt)
            return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    model = CustomBoringModel()
    model.trainer = Trainer()
    with pytest.raises(TypeError, match="provided lr scheduler `CustomScheduler` is invalid"):
        _init_optimizers_and_lr_schedulers(model)


@pytest.mark.parametrize("override", (False, True))
def test_invalid_lr_scheduler_with_custom_step_method(override):
    """Test that custom lr scheduler raises an error if it doesn't follow PyTorch LR Scheduler API."""

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self, foobar):  # breaks the API, forces user to override `lr_scheduler_step`
            ...

        def state_dict(self):
            ...

        def load_state_dict(self, state_dict):
            ...

    class CustomBoringModel(BoringModel):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=1e-2)
            lr_scheduler = CustomScheduler(opt)
            return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    model = CustomBoringModel()
    model.trainer = Trainer()
    if override:

        def lr_scheduler_step(*_):
            ...

        # the user did override the hook, no error
        model.lr_scheduler_step = lr_scheduler_step
        _init_optimizers_and_lr_schedulers(model)
    else:
        with pytest.raises(MisconfigurationException, match="CustomScheduler` doesn't follow"):
            _init_optimizers_and_lr_schedulers(model)
