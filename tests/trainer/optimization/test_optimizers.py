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

import pytest
import torch
from torch import optim

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def test_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        val_check_interval=0.5,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    init_lr = 0.1
    adjusted_lr = [pg['lr'] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_schedulers) == 1
    assert all(a == adjusted_lr[0] for a in adjusted_lr)
    assert init_lr * 0.1 == adjusted_lr[0]


def test_multi_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """

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
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_schedulers) == 2
    assert all(a == adjusted_lr1[0] for a in adjusted_lr1)
    assert all(a == adjusted_lr2[0] for a in adjusted_lr2)
    assert model.init_lr * 0.1 == adjusted_lr1[0]
    assert model.init_lr * 0.1 == adjusted_lr2[0]


def test_reducelronplateau_with_no_monitor_raises(tmpdir):
    """
    Test exception when a ReduceLROnPlateau is used with no monitor
    """
    model = EvalModelTemplate()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: ([optimizer], [optim.lr_scheduler.ReduceLROnPlateau(optimizer)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(
        MisconfigurationException, match='`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`'
    ):
        trainer.fit(model)


def test_reducelronplateau_with_no_monitor_in_lr_scheduler_dict_raises(tmpdir):
    """
    Test exception when lr_scheduler dict has a ReduceLROnPlateau with no monitor
    """
    model = EvalModelTemplate()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        },
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='must include a monitor when a `ReduceLROnPlateau`'):
        trainer.fit(model)


def test_reducelronplateau_scheduling(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters())
            return {
                'optimizer': optimizer,
                'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                'monitor': 'foo',
            }

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    lr_scheduler = trainer.lr_schedulers[0]
    assert lr_scheduler == dict(
        scheduler=lr_scheduler['scheduler'],
        monitor='foo',
        interval='epoch',
        frequency=1,
        reduce_on_plateau=True,
        strict=True,
        opt_idx=None,
        name=None,
    )


def test_optimizer_return_options(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir)
    model = BoringModel()

    # single optimizer
    opt_a = optim.Adam(model.parameters(), lr=0.002)
    opt_b = optim.SGD(model.parameters(), lr=0.002)
    scheduler_a = optim.lr_scheduler.StepLR(opt_a, 10)
    scheduler_b = optim.lr_scheduler.StepLR(opt_b, 10)

    # single optimizer
    model.configure_optimizers = lambda: opt_a
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert len(opt) == 1 and len(lr_sched) == len(freq) == 0

    # opt tuple
    model.configure_optimizers = lambda: (opt_a, opt_b)
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    # opt list
    model.configure_optimizers = lambda: [opt_a, opt_b]
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    ref_lr_sched = dict(
        scheduler=scheduler_a,
        interval='epoch',
        frequency=1,
        reduce_on_plateau=False,
        monitor=None,
        strict=True,
        name=None,
        opt_idx=None,
    )

    # opt tuple of 2 lists
    model.configure_optimizers = lambda: ([opt_a], [scheduler_a])
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt tuple of 1 list
    model.configure_optimizers = lambda: ([opt_a], scheduler_a)
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt single dictionary
    model.configure_optimizers = lambda: {"optimizer": opt_a, "lr_scheduler": scheduler_a}
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert len(opt) == len(lr_sched) == 1
    assert len(freq) == 0
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt multiple dictionaries with frequencies
    model.configure_optimizers = lambda: (
        {
            "optimizer": opt_a,
            "lr_scheduler": scheduler_a,
            "frequency": 1
        },
        {
            "optimizer": opt_b,
            "lr_scheduler": scheduler_b,
            "frequency": 5
        },
    )
    opt, lr_sched, freq = trainer.init_optimizers(model)
    assert len(opt) == len(lr_sched) == len(freq) == 2
    assert opt[0] == opt_a
    ref_lr_sched["opt_idx"] = 0
    assert lr_sched[0] == ref_lr_sched
    ref_lr_sched["scheduler"] = scheduler_b
    ref_lr_sched["opt_idx"] = 1
    assert lr_sched[1] == ref_lr_sched
    assert freq == [1, 5]


def test_none_optimizer(tmpdir):
    model = BoringModel()
    model.configure_optimizers = lambda: None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    with pytest.warns(UserWarning, match='will run with no optimizer'):
        trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_configure_optimizer_from_dict(tmpdir):
    """Tests if `configure_optimizer` method could return a dictionary with `optimizer` field only."""

    class TestModel(BoringModel):

        def configure_optimizers(self):
            config = {'optimizer': optim.SGD(params=self.parameters(), lr=1e-03)}
            return config

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    'schedulers, kwargs, intervals, frequencies, expected_steps, max_epochs',
    [
        (
            (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.OneCycleLR),
            (dict(max_lr=0.01, total_steps=3), dict(max_lr=0.01, total_steps=2)),
            ('step', 'step'),
            (3, 2),
            (4, 3),
            1,
        ),
        (
            (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.OneCycleLR),
            (dict(max_lr=0.01, total_steps=5), dict(max_lr=0.01, total_steps=5)),
            ('step', 'step'),
            (None, None),
            (6, 6),
            1,
        ),
        (
            (optim.lr_scheduler.StepLR, optim.lr_scheduler.CosineAnnealingLR),
            (dict(step_size=5), dict(T_max=2)),
            ('epoch', 'epoch'),
            (5, 10),
            (2, 3),
            3,
        ),
    ],
)
def test_step_scheduling_for_multiple_optimizers_with_frequency(
    tmpdir, schedulers, kwargs, intervals, frequencies, expected_steps, max_epochs
):
    """
    Test that step LR schedulers for multiple optimizers follow
    the optimizer frequencies when corresponding frequency is set.
    """

    class DummyModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs) -> None:
            pass

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.01)
            optimizer2 = optim.Adam(self.parameters(), lr=0.01)

            lr_dict_1 = {
                'scheduler': schedulers[0](optimizer1, **kwargs[0]),
                'interval': intervals[0],
            }
            lr_dict_2 = {
                'scheduler': schedulers[1](optimizer2, **kwargs[1]),
                'interval': intervals[1],
            }

            return [
                {
                    'optimizer': optimizer1,
                    'frequency': frequencies[0],
                    'lr_scheduler': lr_dict_1
                },
                {
                    'optimizer': optimizer2,
                    'frequency': frequencies[1],
                    'lr_scheduler': lr_dict_2
                },
            ]

    model = DummyModel()

    trainer = Trainer(default_root_dir=tmpdir, limit_val_batches=1, limit_train_batches=5, max_epochs=max_epochs)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert trainer.lr_schedulers[0]['opt_idx'] == 0
    assert trainer.lr_schedulers[1]['opt_idx'] == 1
    # Step count is 1 greater than the expected value because scheduler.step() is called once during initialization
    assert trainer.lr_schedulers[0]['scheduler']._step_count == expected_steps[0]
    assert trainer.lr_schedulers[1]['scheduler']._step_count == expected_steps[1]


@pytest.mark.parametrize("fn", ("validate", "test"))
def test_init_optimizers_during_evaluation(tmpdir, fn):
    """
    Test that optimizers is an empty list during evaluation
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.1)
            optimizer2 = optim.Adam(self.parameters(), lr=0.1)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    trainer = Trainer(default_root_dir=tmpdir, limit_val_batches=10, limit_test_batches=10)
    validate_or_test = getattr(trainer, fn)
    validate_or_test(TestModel(), ckpt_path=None)

    assert len(trainer.lr_schedulers) == 0
    assert len(trainer.optimizers) == 0
    assert len(trainer.optimizer_frequencies) == 0


def test_multiple_optimizers_callbacks(tmpdir):
    """
    Tests that multiple optimizers can be used with callbacks
    """

    class CB(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
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
        weights_summary=None,
    )
    trainer.fit(model)


def test_lr_scheduler_strict(tmpdir):
    """
    Test "strict" support in lr_scheduler dict
    """
    model = EvalModelTemplate()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'giraffe',
            'strict': True
        },
    }
    with pytest.raises(
        MisconfigurationException,
        match=r'ReduceLROnPlateau conditioned on metric .* which is not available\. Available metrics are:',
    ):
        trainer.fit(model)

    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'giraffe',
            'strict': False,
        },
    }
    with pytest.warns(
        RuntimeWarning, match=r'ReduceLROnPlateau conditioned on metric .* which is not available but strict'
    ):
        trainer.fit(model)


def test_unknown_configure_optimizers_raises(tmpdir):
    """
    Test exception with an unsupported configure_optimizers return
    """
    model = BoringModel()
    model.configure_optimizers = lambda: 1
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


def test_lr_scheduler_with_unknown_interval_raises(tmpdir):
    """
    Test exception when lr_scheduler dict has unknown interval param value
    """
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, 1),
            'interval': "incorrect_unknown_value"
        },
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match=r'The "interval" key in lr scheduler dict must be'):
        trainer.fit(model)


def test_lr_scheduler_with_extra_keys_warns(tmpdir):
    """
    Test warning when lr_scheduler dict has extra keys
    """
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, 1),
            'foo': 1,
            'bar': 2,
        },
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.warns(RuntimeWarning, match=r'Found unsupported keys in the lr scheduler dict: \[.+\]'):
        trainer.fit(model)


def test_lr_scheduler_with_no_actual_scheduler_raises(tmpdir):
    """
    Test exception when lr_scheduler dict has no scheduler
    """
    model = BoringModel()
    model.configure_optimizers = lambda: {
        'optimizer': optim.Adam(model.parameters()),
        'lr_scheduler': {},
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='The lr scheduler dict must have the key "scheduler"'):
        trainer.fit(model)


def test_invalid_optimizer_in_scheduler(tmpdir):
    """
    Test exception when optimizer attatched to lr_schedulers wasn't returned
    """

    class InvalidOptimizerModel(BoringModel):

        def configure_optimizers(self):
            opt1 = optim.SGD(self.layer.parameters(), lr=0.1)
            opt2 = optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = optim.lr_scheduler.StepLR(opt2, step_size=1)
            return [opt1], [lr_scheduler]

    model = InvalidOptimizerModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="attatched with an optimizer that wasn't returned"):
        trainer.fit(model)


def test_invalid_optimizer_dict_raises(tmpdir):
    """
    Test exception when lr_scheduler dict has no scheduler
    """

    class DummyModel(BoringModel):

        def configure_optimizers(self):
            return [{'optimizer': optim.Adam(self.parameters())}, optim.Adam(self.parameters())]

    model = DummyModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='Unknown configuration for model optimizers'):
        trainer.fit(model)


def test_warn_invalid_scheduler_key_in_manual_optimization(tmpdir):
    """
    Test warning when invalid scheduler keys are provided in manual optimization.
    """

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
    with pytest.warns(RuntimeWarning, match='the keys will be ignored'):
        trainer.fit(model)


@RunIf(min_gpus=2, special=True)
def test_optimizer_state_on_device(tmpdir):
    """ Test that optimizers that create state initially at instantiation still end up with the state on the GPU. """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            # Adagrad creates state tensors immediately, model is not yet on GPU.
            return optim.Adagrad(self.parameters())

        def on_train_start(self, *args, **kwargs):
            opt = self.optimizers()
            _, state = next(iter(opt.state.items()))
            assert state["sum"].device == torch.device("cuda", self.local_rank) == self.device

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=2,
        accelerator="ddp",
        fast_dev_run=True,
    )
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
