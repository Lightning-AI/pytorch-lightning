import os
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

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=None, save_top_k=save_top_k)

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
        self.count = 0
        self.expected_count = expected_count

    def _save_model(self, filepath, trainer, pl_module):
        # make sure we don't save twice
        assert not os.path.isfile(filepath)
        self.count += 1
        super()._save_model(filepath, trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        # on rank 0 we expect the saved files and on all others no saves
        assert (trainer.global_rank == 0 and self.count == self.expected_count) or (
            trainer.global_rank > 0 and self.count == 0
        )


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Distributed training is not supported on Windows",
)
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = EvalModelTemplate()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor='val_loss',
        expected_count=num_epochs, save_top_k=-1
    )
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
    ckpt_name = ModelCheckpoint(monitor='val_loss', filepath=None).format_checkpoint_name(3, {})
    assert ckpt_name == 'epoch=3.ckpt'
    ckpt_name = ModelCheckpoint(monitor='val_loss', filepath='').format_checkpoint_name(5, {})
    assert ckpt_name == 'epoch=5.ckpt'
    # CWD
    ckpt_name = ModelCheckpoint(monitor='val_loss', filepath='.').format_checkpoint_name(3, {})
    assert Path(ckpt_name) == Path('.') / 'epoch=3.ckpt'
    # dir does not exist so it is used as filename
    filepath = tmpdir / 'dir'
    ckpt_name = ModelCheckpoint(monitor='val_loss', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == tmpdir / 'test-dir.ckpt'
    # now, dir exists
    os.mkdir(filepath)
    ckpt_name = ModelCheckpoint(monitor='val_loss', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == filepath / 'test-epoch=3.ckpt'
    # with ver
    ckpt_name = ModelCheckpoint(monitor='val_loss',
                                filepath=tmpdir / 'name', prefix='test').format_checkpoint_name(3, {}, ver=3)
    assert ckpt_name == tmpdir / 'test-name-v3.ckpt'


def test_model_checkpoint_save_last(tmpdir):
    """Tests that save_last produces only one last checkpoint."""
    model = EvalModelTemplate()
    epochs = 3
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last-{epoch}'
    model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=epochs,
    )
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(ModelCheckpoint.CHECKPOINT_NAME_LAST, epochs - 1, {})
    last_filename = last_filename + '.ckpt'
    assert str(tmpdir / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmpdir)) == \
           set([f'epoch={i}.ckpt' for i in range(epochs)] + [last_filename, 'lightning_logs'])
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last'


def test_none_monitor_top_k(tmpdir):
    """
    Make sure that when saving top k of anything (if it's not 1), then monitor cannot be none
    """
    seed_everything(100)
    num_epochs = 3
    with pytest.raises(MisconfigurationException, match=r'To save checkpoints for a top_k metric.*'):
        ModelCheckpoint(filepath=tmpdir, save_top_k=num_epochs, save_last=True)


def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    """Tests that the save_last checkpoint contains the latest information."""
    seed_everything(100)
    model = EvalModelTemplate()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=tmpdir, save_top_k=num_epochs, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=num_epochs,
    )
    trainer.fit(model)

    path_last_epoch = model_checkpoint.format_checkpoint_name(num_epochs - 1, {})
    assert path_last_epoch != model_checkpoint.last_model_path

    ckpt_last_epoch = torch.load(path_last_epoch)
    ckpt_last = torch.load(model_checkpoint.last_model_path)
    assert all(ckpt_last_epoch[k] == ckpt_last[k] for k in ("epoch", "global_step"))
    assert all(
        ckpt_last["callbacks"][type(model_checkpoint)][k] == ckpt_last_epoch["callbacks"][type(model_checkpoint)][k]
        for k in ("best_model_score", "best_model_path")
    )

    # it is easier to load the model objects than to iterate over the raw dict of tensors
    model_last_epoch = EvalModelTemplate.load_from_checkpoint(path_last_epoch)
    model_last = EvalModelTemplate.load_from_checkpoint(model_checkpoint.last_model_path)
    for w0, w1 in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()


def test_model_checkpoint_none_monitor(tmpdir):
    model = EvalModelTemplate()
    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filepath=tmpdir, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
    )
    trainer.fit(model)

    # these should not be set if monitor is None
    assert checkpoint_callback.best_model_path == ''
    assert checkpoint_callback.best_model_score == 0
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''

    # check that the correct ckpts were created
    expected = ['lightning_logs']
    expected.extend(f'epoch={e}.ckpt' for e in range(epochs))
    assert set(os.listdir(tmpdir)) == set(expected)


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
        checkpoint_callback=ModelCheckpoint(monitor='val_loss', filepath=tmpdir + "/{val_loss:.2f}"),
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


def test_default_checkpoint_behavior(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        progress_bar_refresh_rate=0,
        limit_train_batches=5,
        limit_val_batches=5,
    )

    trainer.fit(model)

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
        checkpoint_callback=ModelCheckpoint(monitor='val_loss', filepath=tmpdir + "/{val_loss:.2f}"),
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
        checkpoint_callback=ModelCheckpoint(monitor='val_loss', filepath=tmpdir, save_top_k=0, save_last=save_last),
        max_epochs=max_epochs,
    )
    trainer.fit(model)
    assert caplog.messages.count('Saving latest checkpoint...') == save_last
