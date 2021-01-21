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
from torch import optim

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, EvalModelTemplate


def test_lr_monitor_single_lr(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__single_scheduler

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_monitor],
    )
    result = trainer.fit(model)
    assert result

    assert lr_monitor.lrs, 'No learning rates logged'
    assert all(v is None for v in lr_monitor.last_momentum_values.values()), \
        'Momentum should not be logged by default'
    assert len(lr_monitor.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert lr_monitor.lr_sch_names == list(lr_monitor.lrs.keys()) == ['lr-Adam'], \
        'Names of learning rates not set correctly'


@pytest.mark.parametrize('opt', ['SGD', 'Adam'])
def test_lr_monitor_single_lr_with_momentum(tmpdir, opt):
    """
    Test that learning rates and momentum are extracted and logged for single lr scheduler.
    """
    class LogMomentumModel(BoringModel):
        def __init__(self, opt):
            super().__init__()
            self.opt = opt

        def configure_optimizers(self):
            if self.opt == 'SGD':
                opt_kwargs = {'momentum': 0.9}
            elif self.opt == 'Adam':
                opt_kwargs = {'betas': (0.9, 0.999)}

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
    result = trainer.fit(model)
    assert result

    assert all(v is not None for v in lr_monitor.last_momentum_values.values()), \
        'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_schedulers), \
        'Number of momentum values logged does not match number of lr schedulers'
    assert all(k == f'lr-{opt}-momentum' for k in lr_monitor.last_momentum_values.keys()), \
        'Names of momentum values not set correctly'


def test_log_momentum_no_momentum_optimizer(tmpdir):
    """
    Test that if optimizer doesn't have momentum then a warning is raised with log_momentum=True.
    """
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
        result = trainer.fit(model)
        assert result

    assert all(v == 0 for v in lr_monitor.last_momentum_values.values()), \
        'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_schedulers), \
        'Number of momentum values logged does not match number of lr schedulers'
    assert all(k == 'lr-ASGD-momentum' for k in lr_monitor.last_momentum_values.keys()), \
        'Names of momentum values not set correctly'


def test_lr_monitor_no_lr_scheduler(tmpdir):
    tutils.reset_seed()

    model = EvalModelTemplate()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_monitor],
    )

    with pytest.warns(RuntimeWarning, match='have no learning rate schedulers'):
        result = trainer.fit(model)
        assert result


def test_lr_monitor_no_logger(tmpdir):
    tutils.reset_seed()

    model = EvalModelTemplate()

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[lr_monitor],
        logger=False
    )

    with pytest.raises(MisconfigurationException, match='`Trainer` that has no logger'):
        trainer.fit(model)


@pytest.mark.parametrize("logging_interval", ['step', 'epoch'])
def test_lr_monitor_multi_lrs(tmpdir, logging_interval):
    """ Test that learning rates are extracted and logged for multi lr schedulers. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

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
    result = trainer.fit(model)
    assert result

    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert lr_monitor.lr_sch_names == ['lr-Adam', 'lr-Adam-1'], \
        'Names of learning rates not set correctly'

    if logging_interval == 'step':
        expected_number_logged = trainer.global_step // log_every_n_steps
    if logging_interval == 'epoch':
        expected_number_logged = trainer.max_epochs

    assert all(len(lr) == expected_number_logged for lr in lr_monitor.lrs.values()), \
        'Length of logged learning rates do not match the expected number'


def test_lr_monitor_param_groups(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__param_groups

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_monitor],
    )
    result = trainer.fit(model)
    assert result

    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == 2 * len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of param groups'
    assert lr_monitor.lr_sch_names == ['lr-Adam']
    assert list(lr_monitor.lrs.keys()) == ['lr-Adam/pg1', 'lr-Adam/pg2'], \
        'Names of learning rates not set correctly'


def test_lr_monitor_custom_name(tmpdir):
    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer, [scheduler] = super().configure_optimizers()
            lr_scheduler = {'scheduler': scheduler, 'name': 'my_logging_name'}
            return optimizer, [lr_scheduler]

    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_monitor],
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    trainer.fit(TestModel())
    assert lr_monitor.lr_sch_names == list(lr_monitor.lrs.keys()) == ['my_logging_name']
