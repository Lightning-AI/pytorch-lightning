import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, HyperTuner
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    before_state_dict = model.state_dict()

    tuner = HyperTuner(trainer)
    tuner.scale_batch_size(model, max_trials=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after scaling batch size'


def test_trainer_reset_correctly(tmpdir):
    """ Check that all trainer parameters are reset correctly after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    tuner = HyperTuner(trainer)

    changed_attributes = ['max_steps',
                          'weights_summary',
                          'logger',
                          'callbacks',
                          'checkpoint_callback',
                          'early_stop_callback',
                          'limit_train_batches']

    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    tuner.scale_batch_size(model, max_trials=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


@pytest.mark.parametrize('tuner_arg', [True, 'my_batch_arg'])
def test_tuner_arg(tmpdir, tuner_arg):
    """ Check that tuner arg works with bool input. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()

    class CurrentModel(EvalModelTemplate):
        # Workaround to test if this also works with non-default field input
        @property
        def my_batch_arg(self):
            return self.batch_size

    model = EvalModelTemplate(**hparams)

    before_batch_size = hparams.get('batch_size')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    model.__dict__['my_batch_arg'] = 2
    tuner = HyperTuner(trainer, auto_scale_batch_size=tuner_arg)

    tuner.tune(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


@pytest.mark.parametrize('scale_method', ['power', 'binsearch'])
def test_call_to_tuner_method(tmpdir, scale_method):
    """ Test that calling the tuner method itself works. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_batch_size = hparams.get('batch_size')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )
    tuner = HyperTuner(trainer)

    batch_scaler = tuner.scale_batch_size(model, mode=scale_method, max_trials=8)
    after_batch_size = batch_scaler.suggestion()
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


def test_error_on_dataloader_passed_to_fit(tmpdir):
    """Verify that when the auto scale batch size feature raises an error
       if a train dataloader is passed to fit """

    # only train passed to fit
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    tuner = HyperTuner(trainer, auto_scale_batch_size=True)
    tune_options = dict(train_dataloader=model.dataloader(train=True))

    with pytest.raises(MisconfigurationException):
        tuner.tune(model, **tune_options)


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
    before_batch_size = hparams.get('batch_size')

    trainer = Trainer(**trainer_options)
    tuner = HyperTuner(trainer)
    batch_scaler = tuner.scale_batch_size(model, max_trials=10)
    after_batch_size = batch_scaler.suggestion()

    assert before_batch_size != after_batch_size, \
        'Learning rate was not altered after running learning rate finder'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_amp_support(tmpdir):
    """ Make sure AMP works with learning rate finder """
    tutils.set_random_master_port()

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    before_lr = hparams.get('learning_rate')

    trainer_options = Trainer(
        default_root_dir=tmpdir,
        precision=16,
        gpus=1
    )
    trainer = Trainer(**trainer_options)
    tuner = HyperTuner(trainer)
    lrfinder = tuner.scale_batch_size(model, max_trials=10)
    after_lr = lrfinder.suggestion()

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'
