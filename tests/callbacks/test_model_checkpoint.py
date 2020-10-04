import os
from unittest.mock import MagicMock, Mock

import yaml
import pickle
import platform
import re
from pathlib import Path

import cloudpickle
import pytest
import torch

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate
from pytorch_lightning.utilities.exceptions import MisconfigurationException


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k):
    """Test that None in checkpoint callback is valid and that ckpt_path is set correctly"""
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(monitor='early_stop_on', filepath=None, save_top_k=save_top_k)

    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint,
        overfit_batches=0.20,
        max_epochs=2,
    )
    trainer.fit(model)
    assert (
        checkpoint.dirpath == tmpdir / trainer.logger.name / "version_0" / "checkpoints"
    )


@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_to_yaml(tmpdir, save_top_k):
    """ Test that None in checkpoint callback is valid and that chkp_path is set correctly """
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(filepath=tmpdir, monitor='early_stop_on', save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, checkpoint_callback=checkpoint, overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)

    path_yaml = os.path.join(tmpdir, 'best_k_models.yaml')
    checkpoint.to_yaml(path_yaml)
    d = yaml.full_load(open(path_yaml, 'r'))
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    assert d == best_k


@pytest.mark.parametrize(
    "logger_version,expected",
    [(None, "version_0"), (1, "version_1"), ("awesome", "awesome")],
)
def test_model_checkpoint_path(tmpdir, logger_version, expected):
    """Test that "version_" prefix is only added when logger's version is an integer"""
    tutils.reset_seed()
    model = EvalModelTemplate()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)

    trainer = Trainer(
        default_root_dir=tmpdir, overfit_batches=0.2, max_epochs=2, logger=logger
    )
    trainer.fit(model)

    ckpt_version = Path(trainer.checkpoint_callback.dirpath).parent.name
    assert ckpt_version == expected


def test_pickling(tmpdir):
    ckpt = ModelCheckpoint(tmpdir)

    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)

    ckpt_pickled = cloudpickle.dumps(ckpt)
    ckpt_loaded = cloudpickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)


class ModelCheckpointTestInvocations(ModelCheckpoint):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def __init__(self, expected_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_count = expected_count
        self.on_save_checkpoint_count = 0

    def on_train_start(self, trainer, pl_module):
        torch.save = Mock(wraps=torch.save)

    def on_save_checkpoint(self, trainer, pl_module):
        # expect all ranks to run but only rank 0 will actually write the checkpoint file
        super().on_save_checkpoint(trainer, pl_module)
        self.on_save_checkpoint_count += 1

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        assert self.best_model_path
        assert self.best_model_score
        assert self.on_save_checkpoint_count == self.expected_count
        if trainer.is_global_zero:
            assert torch.save.call_count == self.expected_count
        else:
            assert torch.save.call_count == 0


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Distributed training is not supported on Windows",
)
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = EvalModelTemplate()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor='early_stop_on', expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(
        distributed_backend="ddp_cpu",
        num_processes=2,
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=num_epochs,
    )
    result = trainer.fit(model)
    assert 1 == result


def test_model_checkpoint_format_checkpoint_name(tmpdir):
    # empty filename:
    ckpt_name = ModelCheckpoint._format_checkpoint_name('', 3, {})
    assert ckpt_name == 'epoch=3'
    ckpt_name = ModelCheckpoint._format_checkpoint_name(None, 3, {}, prefix='test')
    assert ckpt_name == 'test-epoch=3'
    # no groups case:
    ckpt_name = ModelCheckpoint._format_checkpoint_name('ckpt', 3, {}, prefix='test')
    assert ckpt_name == 'test-ckpt'
    # no prefix
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch:03d}-{acc}', 3, {'acc': 0.03})
    assert ckpt_name == 'epoch=003-acc=0.03'
    # prefix
    char_org = ModelCheckpoint.CHECKPOINT_JOIN_CHAR
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = '@'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch},{acc:.5f}', 3, {'acc': 0.03}, prefix='test')
    assert ckpt_name == 'test@epoch=3,acc=0.03000'
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = char_org
    # no filepath set
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath=None).format_checkpoint_name(3, {})
    assert ckpt_name == 'epoch=3.ckpt'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath='').format_checkpoint_name(5, {})
    assert ckpt_name == 'epoch=5.ckpt'
    # CWD
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath='.').format_checkpoint_name(3, {})
    assert Path(ckpt_name) == Path('.') / 'epoch=3.ckpt'
    # dir does not exist so it is used as filename
    filepath = tmpdir / 'dir'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == tmpdir / 'test-dir.ckpt'
    # now, dir exists
    os.mkdir(filepath)
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == filepath / 'test-epoch=3.ckpt'
    # with ver
    ckpt_name = ModelCheckpoint(monitor='early_stop_on',
                                filepath=tmpdir / 'name', prefix='test').format_checkpoint_name(3, {}, ver=3)
    assert ckpt_name == tmpdir / 'test-name-v3.ckpt'


def test_model_checkpoint_save_last(tmpdir):
    """Tests that save_last produces only one last checkpoint."""
    model = EvalModelTemplate()
    epochs = 3
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last-{epoch}'
    model_checkpoint = ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(ModelCheckpoint.CHECKPOINT_NAME_LAST, epochs - 1, {})
    last_filename = last_filename + '.ckpt'
    assert str(tmpdir / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmpdir)) == set([f'epoch={i}.ckpt' for i in range(epochs)] + [last_filename])
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last'


def test_invalid_top_k(tmpdir):
    """ Make sure that a MisconfigurationException is raised for a negative save_top_k argument. """
    with pytest.raises(MisconfigurationException, match=r'.*Must be None or >= -1'):
        ModelCheckpoint(filepath=tmpdir, save_top_k=-3)


def test_none_monitor_top_k(tmpdir):
    """ Test that a warning appears for positive top_k with monitor=None. """
    with pytest.raises(
        MisconfigurationException, match=r'ModelCheckpoint\(save_top_k=3, monitor=None\) is not a valid*'
    ):
        ModelCheckpoint(filepath=tmpdir, save_top_k=3)
    # These should not fail
    ModelCheckpoint(filepath=tmpdir, save_top_k=None)
    ModelCheckpoint(filepath=tmpdir, save_top_k=-1)
    ModelCheckpoint(filepath=tmpdir, save_top_k=0)


def test_none_monitor_save_last(tmpdir):
    """ Test that a warning appears for save_last=True with monitor=None. """
    with pytest.raises(
        MisconfigurationException, match=r'ModelCheckpoint\(save_last=True, monitor=None\) is not a valid.*'
    ):
        ModelCheckpoint(filepath=tmpdir, save_last=True)
    # These should not fail
    ModelCheckpoint(filepath=tmpdir, save_last=None)
    ModelCheckpoint(filepath=tmpdir, save_last=False)


def test_model_checkpoint_none_monitor(tmpdir):
    """ Test that it is possible to save all checkpoints when monitor=None. """
    model = EvalModelTemplate()
    model.validation_step = model.validation_step_no_monitor
    model.validation_epoch_end = model.validation_epoch_end_no_monitor

    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, filepath=tmpdir, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)

    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == checkpoint_callback.last_model_path == tmpdir / 'epoch=1.ckpt'
    assert checkpoint_callback.best_model_score == 0
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''

    # check that the correct ckpts were created
    expected = [f'epoch={e}.ckpt' for e in range(epochs)]
    assert set(os.listdir(tmpdir)) == set(expected)


@pytest.mark.parametrize("period", list(range(4)))
def test_model_checkpoint_period(tmpdir, period):
    model = EvalModelTemplate()
    epochs = 5
    checkpoint_callback = ModelCheckpoint(filepath=tmpdir, save_top_k=-1, period=period)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        logger=False,
    )
    trainer.fit(model)

    # check that the correct ckpts were created
    expected = [f'epoch={e}.ckpt' for e in range(epochs) if not (e + 1) % period] if period > 0 else []
    assert set(os.listdir(tmpdir)) == set(expected)


def test_model_checkpoint_topk_zero(tmpdir):
    """ Test that no checkpoints are saved when save_top_k=0. """
    model = EvalModelTemplate()
    checkpoint_callback = ModelCheckpoint(filepath=tmpdir, save_top_k=0)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=2,
        logger=False,
    )
    trainer.fit(model)
    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == ''
    assert checkpoint_callback.best_model_score == 0
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''
    # check that no ckpts were created
    assert len(os.listdir(tmpdir)) == 0


def test_model_checkpoint_topk_all(tmpdir):
    """ Test that save_top_k=-1 tracks the best models when monitor key is provided. """
    seed_everything(1000)
    epochs = 2
    model = EvalModelTemplate()
    checkpoint_callback = ModelCheckpoint(filepath=tmpdir, monitor="early_stop_on", save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)
    assert checkpoint_callback.best_model_path == tmpdir / "epoch=1.ckpt"
    assert checkpoint_callback.best_model_score > 0
    assert set(checkpoint_callback.best_k_models.keys()) == set(str(tmpdir / f"epoch={i}.ckpt") for i in range(epochs))
    assert checkpoint_callback.kth_best_model_path == tmpdir / "epoch=0.ckpt"


def test_ckpt_metric_names(tmpdir):
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        progress_bar_refresh_rate=0,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir + "/{val_loss:.2f}"),
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


def test_default_checkpoint_behavior(tmpdir):
    seed_everything(1234)

    os.environ['PL_DEV_DEBUG'] = '1'
    model = EvalModelTemplate()
    model.validation_step = model.validation_step_no_monitor
    model.validation_epoch_end = model.validation_epoch_end_no_monitor

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        progress_bar_refresh_rate=0,
        limit_train_batches=5,
        limit_val_batches=5,
    )

    trainer.fit(model)
    results = trainer.test()

    assert len(results) == 1
    assert results[0]['test_acc'] >= 0.80
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 3

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(os.path.join(tmpdir, 'lightning_logs', 'version_0', 'checkpoints'))
    assert len(ckpts) == 1
    assert ckpts[0] == 'epoch=2.ckpt'


def test_ckpt_metric_names_results(tmpdir):
    model = EvalModelTemplate()
    model.training_step = model.training_step_result_obj
    model.training_step_end = None
    model.training_epoch_end = None

    model.validation_step = model.validation_step_result_obj
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        progress_bar_refresh_rate=0,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir + "/{val_loss:.2f}"),
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


@pytest.mark.parametrize('max_epochs', [1, 2])
@pytest.mark.parametrize('should_validate', [True, False])
@pytest.mark.parametrize('save_last', [True, False])
def test_model_checkpoint_save_last_warning(tmpdir, caplog, max_epochs, should_validate, save_last):
    """Tests 'Saving latest checkpoint...' log"""
    model = EvalModelTemplate()
    if not should_validate:
        model.validation_step = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir, save_top_k=0, save_last=save_last),
        max_epochs=max_epochs,
    )
    trainer.fit(model)
    assert caplog.messages.count('Saving latest checkpoint...') == save_last


def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    """ Tests that the save_last checkpoint contains the latest information. """
    seed_everything(100)
    model = EvalModelTemplate()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(
        monitor='early_stop_on', filepath=tmpdir, save_top_k=num_epochs, save_last=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=num_epochs,
    )
    trainer.fit(model)

    path_last_epoch = str(tmpdir / f"epoch={num_epochs - 1}.ckpt")
    path_last = str(tmpdir / "last.ckpt")
    assert path_last == model_checkpoint.last_model_path

    ckpt_last_epoch = torch.load(path_last_epoch)
    ckpt_last = torch.load(path_last)
    assert all(ckpt_last_epoch[k] == ckpt_last[k] for k in ("epoch", "global_step"))

    ch_type = type(model_checkpoint)
    assert all(list(
        ckpt_last["callbacks"][ch_type][k] == ckpt_last_epoch["callbacks"][ch_type][k]
        for k in ("best_model_score", "best_model_path")
    ))

    # it is easier to load the model objects than to iterate over the raw dict of tensors
    model_last_epoch = EvalModelTemplate.load_from_checkpoint(path_last_epoch)
    model_last = EvalModelTemplate.load_from_checkpoint(
        model_checkpoint.last_model_path
    )
    for w0, w1 in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()


def test_model_checkpoint_only_weights(tmpdir):
    """Tests use case where ModelCheckpoint is configured to save only model weights, and
     user tries to load checkpoint to resume training.
     """
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, monitor='early_stop_on', save_weights_only=True),
    )
    # fit model
    result = trainer.fit(model)
    # training complete
    assert result == 1, 'training failed to complete'

    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]

    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(checkpoint_path)
    assert 'optimizer_states' not in checkpoint, 'checkpoint should contain only model weights'
    assert 'lr_schedulers' not in checkpoint, 'checkpoint should contain only model weights'

    # assert loading model works when checkpoint has only weights
    assert EvalModelTemplate.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # directly save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path, weights_only=True)
    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(new_weights_path)
    assert 'optimizer_states' not in checkpoint, 'checkpoint should contain only model weights'
    assert 'lr_schedulers' not in checkpoint, 'checkpoint should contain only model weights'

    # assert restoring train state fails
    with pytest.raises(KeyError, match='checkpoint contains only the model'):
        trainer.checkpoint_connector.restore_training_state(checkpoint)


@pytest.mark.parametrize(["save_top_k", "save_last", "file_prefix", "expected_files"], [
    pytest.param(-1, False, '', {'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt', 'epoch=1.ckpt', 'epoch=0.ckpt'},
                 id="CASE K=-1  (all)"),
    pytest.param(1, False, 'test_prefix', {'test_prefix-epoch=4.ckpt'},
                 id="CASE K=1 (2.5, epoch 4)"),
    pytest.param(2, False, '', {'epoch=4.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
    pytest.param(4, False, '', {'epoch=1.ckpt', 'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=4 (save all 4 base)"),
    pytest.param(3, False, '', {'epoch=2.ckpt', 'epoch=3.ckpt', 'epoch=4.ckpt'},
                 id="CASE K=3 (save the 2nd, 3rd, 4th model)"),
    pytest.param(1, True, '', {'epoch=4.ckpt', 'last.ckpt'},
                 id="CASE K=1 (save the 4th model and the last model)"),
])
def test_model_checkpoint_options(tmpdir, save_top_k, save_last, file_prefix, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath, *args):
        open(filepath, 'a').close()

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(
        tmpdir, monitor='checkpoint_on', save_top_k=save_top_k, save_last=save_last, prefix=file_prefix, verbose=1
    )
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.global_step = i
        trainer.logger_connector.callback_metrics = {'checkpoint_on': torch.tensor(loss)}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(tmpdir))

    assert len(file_lists) == len(expected_files), (
        f"Should save {len(expected_files)} models when save_top_k={save_top_k} but found={file_lists}"
    )

    # verify correct naming
    for fname in expected_files:
        assert fname in file_lists
