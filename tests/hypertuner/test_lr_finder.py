import pytest
import torch

import tests.base.develop_utils as tutils
from tests.base import EvalModelTemplate
from pytorch_lightning import Trainer, HyperTuner
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_error_on_more_than_1_optimizer(tmpdir):
    """ Check that error is thrown when more than 1 optimizer is passed """

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    tuner = HyperTuner(trainer)

    with pytest.raises(MisconfigurationException):
        tuner.lr_find(model)


def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after lr_find() """

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    tuner = HyperTuner(trainer)

    before_state_dict = model.state_dict()

    _ = tuner.lr_find(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after learning rate finder'


def test_trainer_reset_correctly(tmpdir):
    """ Check that all trainer parameters are reset correctly after lr_find() """

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    tuner = HyperTuner(trainer)

    changed_attributes = ['callbacks', 'logger', 'val_check_interval',
                          'max_steps', 'checkpoint_callback', 'early_stop_callback']
    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)
    attributes_before['configure_optimizers'] = getattr(model, 'configure_optimizers')

    _ = tuner.lr_find(model, num_training=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)
    attributes_after['configure_optimizers'] = getattr(model, 'configure_optimizers')

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


def test_tuner_arg_bool(tmpdir):
    """ Test that setting tuner arg to bool works """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    before_lr = hparams.get('learning_rate')

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_lr_find=True,
    )
    tuner = HyperTuner(trainer, auto_lr_find=True)

    tuner.tune(model)
    after_lr = model.learning_rate
    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


def test_tuner_arg_str(tmpdir):
    """ Test that setting tuner arg to string works """
    model = EvalModelTemplate()
    model.my_fancy_lr = 1.0  # update with non-standard field

    before_lr = model.my_fancy_lr
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_lr_find='my_fancy_lr',
    )
    tuner = HyperTuner(trainer, auto_lr_find='my_fancy_lr')

    tuner.tune(model)
    after_lr = model.my_fancy_lr
    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


def test_call_to_tuner_method(tmpdir):
    """ Test that directly calling the tuner method works """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_lr = hparams.get('learning_rate')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
    )
    tuner = HyperTuner(trainer)

    lrfinder = tuner.lr_find(model, mode='linear')
    after_lr = lrfinder.suggestion()
    model.learning_rate = after_lr

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
    tuner = HyperTuner(trainer)

    lrfinder = tuner.lr_find(model, num_training=20, early_stop_threshold=None)
    after_lr = lrfinder.suggestion()

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'
    assert len(lrfinder.results['lr']) == 20, \
        'Early stopping for learning rate finder did not work'
    assert lrfinder._total_batch_idx == 32, \
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
    tuner = HyperTuner(trainer)

    lrfinder = tuner.lr_find(model)
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
    tuner = HyperTuner(trainer)

    lrfinder = tuner.lr_find(model)
    before_lr = lrfinder.suggestion()
    lrfinder.results['loss'][-1] = float('nan')
    after_lr = lrfinder.suggestion()

    assert before_lr == after_lr, \
        'Learning rate was altered because of non-finite loss values'


@pytest.mark.spawn
@pytest.mark.parametrize("backend", ['dp', 'ddp', 'ddp2'])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model(tmpdir, backend):
    """Make sure DDP works."""
    tutils.set_random_master_port()

    trainer_options = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        gpus=[0, 1],
        distributed_backend=backend
    )

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    before_lr = hparams.get('learning_rate')

    trainer = Trainer(**trainer_options)
    tuner = HyperTuner(trainer)
    lrfinder = tuner.lr_find(model)
    after_lr = lrfinder.suggestion()

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'
