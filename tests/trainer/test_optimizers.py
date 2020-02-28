import math
import os

import pytest
import torch

import tests.models.utils as tutils
from pytorch_lightning import Trainer

from tests.models import (
    TestModelBase,
    LightTrainDataloader,
    LightTestOptimizerWithSchedulingMixin,
    LightTestMultipleOptimizersWithSchedulingMixin,
    LightTestOptimizersWithMixedSchedulingMixin
)


def test_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """
    tutils.reset_seed()

    class CurrentTestModel(
            LightTestOptimizerWithSchedulingMixin,
            LightTrainDataloader,
            TestModelBase):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    results = trainer.fit(model)

    init_lr = hparams.learning_rate
    adjusted_lr = [pg['lr'] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_schedulers) == 1, \
        'lr scheduler not initialized properly'

    assert all(a == adjusted_lr[0] for a in adjusted_lr), \
        'lr not equally adjusted for all param groups'
    adjusted_lr = adjusted_lr[0]

    assert init_lr * 0.1 == adjusted_lr, \
        'lr not adjusted correctly'


def test_multi_optimizer_with_scheduling(tmpdir):
    """ Verify that learning rate scheduling is working """
    tutils.reset_seed()

    class CurrentTestModel(
            LightTestMultipleOptimizersWithSchedulingMixin,
            LightTrainDataloader,
            TestModelBase):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    results = trainer.fit(model)

    init_lr = hparams.learning_rate
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

    assert init_lr * 0.1 == adjusted_lr1 and init_lr * 0.1 == adjusted_lr2, \
        'lr not adjusted correctly'


def test_multi_optimizer_with_scheduling_stepping(tmpdir):
    tutils.reset_seed()

    class CurrentTestModel(
            LightTestOptimizersWithMixedSchedulingMixin,
            LightTrainDataloader,
            TestModelBase):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    results = trainer.fit(model)

    init_lr = hparams.learning_rate
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
    assert init_lr * (0.1)**3 == adjusted_lr1, \
        'lr for optimizer 1 not adjusted correctly'
    # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times
    assert init_lr * 0.1 == adjusted_lr2, \
        'lr for optimizer 2 not adjusted correctly'


# if __name__ == '__main__':
#     pytest.main([__file__])
