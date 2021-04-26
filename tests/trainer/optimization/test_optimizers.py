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

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.helpers.boring_model import BoringModel


def test_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__single_scheduler

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        val_check_interval=0.5,
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    init_lr = hparams.get('learning_rate')
    adjusted_lr = [pg['lr'] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_schedulers) == 1, \
        'lr scheduler not initialized properly, it has %i elements instread of 1' % len(trainer.lr_schedulers)

    assert all(a == adjusted_lr[0] for a in adjusted_lr), \
        'Lr not equally adjusted for all param groups'
    adjusted_lr = adjusted_lr[0]

    assert init_lr * 0.1 == adjusted_lr, \
        'Lr not adjusted correctly, expected %f but got %f' % (init_lr * 0.1, adjusted_lr)


def test_multi_optimizer_with_scheduling_stepping(tmpdir):

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    init_lr = hparams.get('learning_rate')
    adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_schedulers) == 2, 'all lr scheduler not initialized properly'

    assert all(a == adjusted_lr1[0] for a in adjusted_lr1), \
        'lr not equally adjusted for all param groups for optimizer 1'
    adjusted_lr1 = adjusted_lr1[0]

    assert all(a == adjusted_lr2[0] for a in adjusted_lr2), \
        'lr not equally adjusted for all param groups for optimizer 2'
    adjusted_lr2 = adjusted_lr2[0]

    # Called ones after end of epoch
    assert init_lr * 0.1 == adjusted_lr1, 'lr for optimizer 1 not adjusted correctly'
    # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times
    assert init_lr * 0.1 == adjusted_lr2, 'lr for optimizer 2 not adjusted correctly'


def test_reducelronplateau_with_no_monitor_raises(tmpdir):
    """
    Test exception when a ReduceLROnPlateau is used with no monitor
    """
    model = EvalModelTemplate()
    optimizer = torch.optim.Adam(model.parameters())
    model.configure_optimizers = lambda: ([optimizer], [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)])
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
    optimizer = torch.optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        },
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='must include a monitor when a `ReduceLROnPlateau`'):
        trainer.fit(model)


def test_reducelronplateau_scheduling(tmpdir):
    model = EvalModelTemplate()
    optimizer = torch.optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        'monitor': 'val_acc',
    }
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    lr_scheduler = trainer.lr_schedulers[0]
    assert lr_scheduler == dict(
        scheduler=lr_scheduler['scheduler'],
        monitor='val_acc',
        interval='epoch',
        frequency=1,
        reduce_on_plateau=True,
        strict=True,
        name=None,
    ), 'lr scheduler was not correctly converted to dict'


def test_optimizer_return_options():
    trainer = Trainer()
    model = EvalModelTemplate()

    # single optimizer
    opt_a = torch.optim.Adam(model.parameters(), lr=0.002)
    opt_b = torch.optim.SGD(model.parameters(), lr=0.002)
    scheduler_a = torch.optim.lr_scheduler.StepLR(opt_a, 10)
    scheduler_b = torch.optim.lr_scheduler.StepLR(opt_b, 10)

    # single optimizer
    model.configure_optimizers = lambda: opt_a
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 1 and len(lr_sched) == len(freq) == 0

    # opt tuple
    model.configure_optimizers = lambda: (opt_a, opt_b)
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert optim == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    # opt list
    model.configure_optimizers = lambda: [opt_a, opt_b]
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert optim == [opt_a, opt_b]
    assert len(lr_sched) == len(freq) == 0

    ref_lr_sched = dict(
        scheduler=scheduler_a,
        interval='epoch',
        frequency=1,
        reduce_on_plateau=False,
        monitor=None,
        strict=True,
        name=None,
    )

    # opt tuple of 2 lists
    model.configure_optimizers = lambda: ([opt_a], [scheduler_a])
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == len(lr_sched) == 1
    assert len(freq) == 0
    assert optim[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt tuple of 1 list
    model.configure_optimizers = lambda: ([opt_a], scheduler_a)
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == len(lr_sched) == 1
    assert len(freq) == 0
    assert optim[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt single dictionary
    model.configure_optimizers = lambda: {"optimizer": opt_a, "lr_scheduler": scheduler_a}
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == len(lr_sched) == 1
    assert len(freq) == 0
    assert optim[0] == opt_a
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
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == len(lr_sched) == len(freq) == 2
    assert optim[0] == opt_a
    assert lr_sched[0] == ref_lr_sched
    assert freq == [1, 5]


def test_none_optimizer_warning():

    trainer = Trainer()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__empty

    with pytest.warns(UserWarning, match='will run with no optimizer'):
        _, __, ___ = trainer.init_optimizers(model)


def test_none_optimizer(tmpdir):

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__empty

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    trainer.fit(model)

    # verify training completed
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


def test_configure_optimizer_from_dict(tmpdir):
    """Tests if `configure_optimizer` method could return a dictionary with `optimizer` field only."""

    class CurrentModel(EvalModelTemplate):

        def configure_optimizers(self):
            config = {'optimizer': torch.optim.SGD(params=self.parameters(), lr=1e-03)}
            return config

    hparams = EvalModelTemplate.get_default_hparams()
    model = CurrentModel(**hparams)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


def test_configure_optimizers_with_frequency(tmpdir):
    """
    Test that multiple optimizers work when corresponding frequency is set.
    """
    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_optimizers_frequency

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


@pytest.mark.parametrize("fn", ("validate", "test"))
def test_init_optimizers_during_evaluation(tmpdir, fn):
    """
    Test that optimizers is an empty list during evaluation
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            optimizer1 = torch.optim.Adam(self.parameters(), lr=0.1)
            optimizer2 = torch.optim.Adam(self.parameters(), lr=0.1)
            lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)
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
            a = torch.optim.RMSprop(self.layer_1.parameters(), 1e-2)
            b = torch.optim.RMSprop(self.layer_2.parameters(), 1e-2)
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
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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
        assert trainer.fit(model)


def test_unknown_configure_optimizers_raises(tmpdir):
    """
    Test exception with an unsupported configure_optimizers return
    """
    model = EvalModelTemplate()
    model.configure_optimizers = lambda: 1
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


def test_lr_scheduler_with_unknown_interval_raises(tmpdir):
    """
    Test exception when lr_scheduler dict has unknown interval param value
    """
    model = BoringModel()
    optimizer = torch.optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, 1),
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
    model = EvalModelTemplate()
    optimizer = torch.optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, 1),
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
    model = EvalModelTemplate()
    model.configure_optimizers = lambda: {
        'optimizer': torch.optim.Adam(model.parameters()),
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
            opt1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            opt2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(opt2, step_size=1)
            return [opt1], [lr_scheduler]

    model = InvalidOptimizerModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="attatched with an optimizer that wasn't returned"):
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
            opt = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return [opt], [{"scheduler": sch, "interval": "epoch"}]

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.warns(RuntimeWarning, match='the keys will be ignored'):
        trainer.fit(model)
