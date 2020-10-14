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

from pytorch_lightning import Trainer, Callback
from tests.base import EvalModelTemplate
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base.boring_model import BoringModel


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
    )
    results = trainer.fit(model)
    assert results == 1

    init_lr = hparams.get('learning_rate')
    adjusted_lr = [pg['lr'] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_schedulers) == 1, \
        'lr scheduler not initialized properly, it has %i elements instread of 1' % len(trainer.lr_schedulers)

    assert all(a == adjusted_lr[0] for a in adjusted_lr), \
        'Lr not equally adjusted for all param groups'
    adjusted_lr = adjusted_lr[0]

    assert init_lr * 0.1 == adjusted_lr, \
        'Lr not adjusted correctly, expected %f but got %f' % (init_lr * 0.1, adjusted_lr)


def test_multi_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """

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
    results = trainer.fit(model)
    assert results == 1

    init_lr = hparams.get('learning_rate')
    adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_schedulers) == 2, \
        'all lr scheduler not initialized properly, it has %i elements instread of 1' % len(trainer.lr_schedulers)

    assert all(a == adjusted_lr1[0] for a in adjusted_lr1), \
        'Lr not equally adjusted for all param groups for optimizer 1'
    adjusted_lr1 = adjusted_lr1[0]

    assert all(a == adjusted_lr2[0] for a in adjusted_lr2), \
        'Lr not equally adjusted for all param groups for optimizer 2'
    adjusted_lr2 = adjusted_lr2[0]

    assert init_lr * 0.1 == adjusted_lr1 and init_lr * 0.1 == adjusted_lr2, \
        'Lr not adjusted correctly, expected %f but got %f' % (init_lr * 0.1, adjusted_lr1)


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
    results = trainer.fit(model)
    assert results == 1

    init_lr = hparams.get('learning_rate')
    adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_schedulers) == 2, \
        'all lr scheduler not initialized properly'

    assert all(a == adjusted_lr1[0] for a in adjusted_lr1), \
        'lr not equally adjusted for all param groups for optimizer 1'
    adjusted_lr1 = adjusted_lr1[0]

    assert all(a == adjusted_lr2[0] for a in adjusted_lr2), \
        'lr not equally adjusted for all param groups for optimizer 2'
    adjusted_lr2 = adjusted_lr2[0]

    # Called ones after end of epoch
    assert init_lr * 0.1 ** 1 == adjusted_lr1, \
        'lr for optimizer 1 not adjusted correctly'
    # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times
    assert init_lr * 0.1 == adjusted_lr2, \
        'lr for optimizer 2 not adjusted correctly'


def test_reduce_lr_on_plateau_scheduling_missing_monitor(tmpdir):

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__reduce_lr_on_plateau

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )

    m = '.*ReduceLROnPlateau requires returning a dict from configure_optimizers.*'
    with pytest.raises(MisconfigurationException, match=m):
        trainer.fit(model)


def test_reduce_lr_on_plateau_scheduling(tmpdir):
    hparams = EvalModelTemplate.get_default_hparams()
    class TestModel(EvalModelTemplate):

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'early_stop_on'}

    model = TestModel(**hparams)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    results = trainer.fit(model)
    assert results == 1

    assert trainer.lr_schedulers[0] == \
        dict(scheduler=trainer.lr_schedulers[0]['scheduler'], monitor='early_stop_on',
             interval='epoch', frequency=1, reduce_on_plateau=True), \
        'lr schduler was not correctly converted to dict'


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
    assert len(optim) == 1 and len(lr_sched) == 0 and len(freq) == 0

    # opt tuple
    model.configure_optimizers = lambda: (opt_a, opt_b)
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 2 and optim[0] == opt_a and optim[1] == opt_b
    assert len(lr_sched) == 0 and len(freq) == 0

    # opt list
    model.configure_optimizers = lambda: [opt_a, opt_b]
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 2 and optim[0] == opt_a and optim[1] == opt_b
    assert len(lr_sched) == 0 and len(freq) == 0

    # opt tuple of 2 lists
    model.configure_optimizers = lambda: ([opt_a], [scheduler_a])
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 1 and len(lr_sched) == 1 and len(freq) == 0
    assert optim[0] == opt_a
    assert lr_sched[0] == dict(scheduler=scheduler_a, interval='epoch',
                               frequency=1, reduce_on_plateau=False)

    # opt single dictionary
    model.configure_optimizers = lambda: {"optimizer": opt_a, "lr_scheduler": scheduler_a}
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 1 and len(lr_sched) == 1 and len(freq) == 0
    assert optim[0] == opt_a
    assert lr_sched[0] == dict(scheduler=scheduler_a, interval='epoch',
                               frequency=1, reduce_on_plateau=False)

    # opt multiple dictionaries with frequencies
    model.configure_optimizers = lambda: (
        {"optimizer": opt_a, "lr_scheduler": scheduler_a, "frequency": 1},
        {"optimizer": opt_b, "lr_scheduler": scheduler_b, "frequency": 5},
    )
    optim, lr_sched, freq = trainer.init_optimizers(model)
    assert len(optim) == 2 and len(lr_sched) == 2 and len(freq) == 2
    assert optim[0] == opt_a
    assert lr_sched[0] == dict(scheduler=scheduler_a, interval='epoch',
                               frequency=1, reduce_on_plateau=False)
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
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


def test_configure_optimizer_from_dict(tmpdir):
    """Tests if `configure_optimizer` method could return a dictionary with `optimizer` field only."""

    class CurrentModel(EvalModelTemplate):
        def configure_optimizers(self):
            config = {
                'optimizer': torch.optim.SGD(params=self.parameters(), lr=1e-03)
            }
            return config

    hparams = EvalModelTemplate.get_default_hparams()
    model = CurrentModel(**hparams)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    result = trainer.fit(model)
    assert result == 1


def test_configure_optimizers_with_frequency(tmpdir):
    """
    Test that multiple optimizers work when corresponding frequency is set.
    """
    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_optimizers_frequency

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    result = trainer.fit(model)
    assert result


def test_init_optimizers_during_testing(tmpdir):
    """
    Test that optimizers is an empty list during testing.
    """
    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_test_batches=10
    )
    trainer.test(model, ckpt_path=None)

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
