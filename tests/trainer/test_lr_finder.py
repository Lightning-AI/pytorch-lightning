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
from copy import deepcopy

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel


def test_error_on_more_than_1_optimizer(tmpdir):
    """ Check that error is thrown when more than 1 optimizer is passed """

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    with pytest.raises(MisconfigurationException):
        trainer.tuner.lr_find(model)


def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after lr_find() """

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    before_state_dict = deepcopy(model.state_dict())

    _ = trainer.tuner.lr_find(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after learning rate finder'

    assert not os.path.exists(tmpdir / 'lr_find_temp_model.ckpt')


def test_trainer_reset_correctly(tmpdir):
    """ Check that all trainer parameters are reset correctly after lr_find() """

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    changed_attributes = [
        'callbacks', 'logger', 'max_steps', 'auto_lr_find', 'accumulate_grad_batches', 'checkpoint_callback'
    ]
    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    _ = trainer.tuner.lr_find(model, num_training=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'

    assert model.trainer == trainer


@pytest.mark.parametrize('use_hparams', [False, True])
def test_trainer_arg_bool(tmpdir, use_hparams):
    """ Test that setting trainer arg to bool works """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    before_lr = hparams.get('learning_rate')
    if use_hparams:
        del model.learning_rate
        model.configure_optimizers = model.configure_optimizers__lr_from_hparams

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_lr_find=True,
    )

    trainer.tune(model)
    if use_hparams:
        after_lr = model.hparams.learning_rate
    else:
        after_lr = model.learning_rate

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


@pytest.mark.parametrize('use_hparams', [False, True])
def test_trainer_arg_str(tmpdir, use_hparams):
    """ Test that setting trainer arg to string works """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.my_fancy_lr = 1.0  # update with non-standard field
    model.hparams['my_fancy_lr'] = 1.0
    before_lr = model.my_fancy_lr
    if use_hparams:
        del model.my_fancy_lr
        model.configure_optimizers = model.configure_optimizers__lr_from_hparams

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_lr_find='my_fancy_lr',
    )

    trainer.tune(model)
    if use_hparams:
        after_lr = model.hparams.my_fancy_lr
    else:
        after_lr = model.my_fancy_lr

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


@pytest.mark.parametrize('optimizer', ['Adam', 'Adagrad'])
def test_call_to_trainer_method(tmpdir, optimizer):
    """ Test that directly calling the trainer method works """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    if optimizer == 'adagrad':
        model.configure_optimizers = model.configure_optimizers__adagrad

    before_lr = hparams.get('learning_rate')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
    )

    lrfinder = trainer.tuner.lr_find(model, mode='linear')
    after_lr = lrfinder.suggestion()
    model.learning_rate = after_lr
    trainer.tune(model)

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


def test_datamodule_parameter(tmpdir):
    """ Test that the datamodule parameter works """

    dm = ClassifDataModule()
    model = ClassificationModel()

    before_lr = model.lr
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
    )

    lrfinder = trainer.tuner.lr_find(model, datamodule=dm)
    after_lr = lrfinder.suggestion()
    model.lr = after_lr

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


def test_accumulation_and_early_stopping(tmpdir):
    """ Test that early stopping of learning rate finder works, and that
        accumulation also works for this feature """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_lr = hparams.get('learning_rate')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        accumulate_grad_batches=2,
    )

    lrfinder = trainer.tuner.lr_find(model, early_stop_threshold=None)
    after_lr = lrfinder.suggestion()

    expected_num_lrs = 100
    expected_batch_idx = 200 - 1

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'
    assert len(lrfinder.results['lr']) == expected_num_lrs, \
        'Early stopping for learning rate finder did not work'
    assert lrfinder._total_batch_idx == expected_batch_idx, \
        'Accumulation parameter did not work'


def test_suggestion_parameters_work(tmpdir):
    """ Test that default skipping does not alter results in basic case """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
    )

    lrfinder = trainer.tuner.lr_find(model)
    lr1 = lrfinder.suggestion(skip_begin=10)  # default
    lr2 = lrfinder.suggestion(skip_begin=80)  # way too high, should have an impact

    assert lr1 != lr2, \
        'Skipping parameter did not influence learning rate'


def test_suggestion_with_non_finite_values(tmpdir):
    """ Test that non-finite values does not alter results """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
    )

    lrfinder = trainer.tuner.lr_find(model)
    before_lr = lrfinder.suggestion()
    lrfinder.results['loss'][-1] = float('nan')
    after_lr = lrfinder.suggestion()

    assert before_lr == after_lr, \
        'Learning rate was altered because of non-finite loss values'


def test_lr_finder_fails_fast_on_bad_config(tmpdir):
    """ Test that tune fails if the model does not have a lr BEFORE running lr find """
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, auto_lr_find=True)
    with pytest.raises(MisconfigurationException, match='should have one of these fields'):
        trainer.tune(BoringModel())
