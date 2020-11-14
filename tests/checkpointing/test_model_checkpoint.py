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
import os.path as osp
import pickle
import platform
import re
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import cloudpickle
import pytest
import torch
import yaml

import pytorch_lightning as pl
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, EvalModelTemplate


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize('save_top_k', [-1])
def test_model_checkpoint_correct_score(tmpdir, save_top_k):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path"""
    tutils.reset_seed()

    model = EvalModelTemplate()

    filename = "{val_acc:.4f}-{epoch}"

    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor='val_acc', save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, checkpoint_callback=checkpoint, overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)

    ckpt_files = list(Path(tmpdir).glob('*.ckpt'))

    metrics = trainer.dev_debugger.logged_metrics
    expected_filenames = {f'val_acc={metric["val_acc"]:.4f}-epoch={metric["epoch"]}.ckpt' for metric in metrics}
    for ckpt_file in ckpt_files:
        assert os.path.basename(ckpt_file) in expected_filenames


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k):
    """Test that dirpath=None in checkpoint callback is valid and that ckpt_path is set correctly"""
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=None, filename='{epoch}', save_top_k=save_top_k)
    max_epochs = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint,
        overfit_batches=0.20,
        max_epochs=max_epochs,
    )
    trainer.fit(model)
    assert (
        checkpoint.dirpath == tmpdir / trainer.logger.name / "version_0" / "checkpoints"
    )

    if save_top_k == -1:
        ckpt_files = os.listdir(checkpoint.dirpath)
        expected_ckpt_files = [f'epoch={i}.ckpt' for i in range(max_epochs)]
        assert len(ckpt_files) == len(expected_ckpt_files) == max_epochs
        assert set(ckpt_files) == set(expected_ckpt_files)


@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_to_yaml(tmpdir, save_top_k):
    """ Test that None in checkpoint callback is valid and that chkp_path is set correctly """
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_top_k=save_top_k)

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
    ckpt = ModelCheckpoint(dirpath=tmpdir)

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

    # no dirpath set
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath=None).format_checkpoint_name(3, {})
    assert ckpt_name == 'epoch=3.ckpt'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='').format_checkpoint_name(5, {})
    assert ckpt_name == 'epoch=5.ckpt'

    # CWD
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='.').format_checkpoint_name(3, {})
    assert ckpt_name == str(Path('.').resolve() / 'epoch=3.ckpt')

    # with ver
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', dirpath=tmpdir, filename='name', prefix='test'
    ).format_checkpoint_name(3, {}, ver=3)
    assert ckpt_name == tmpdir / 'test-name-v3.ckpt'

    # using slashes
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', dirpath=None, filename='{epoch}_{val/loss:.5f}'
    ).format_checkpoint_name(4, {'val/loss': 0.03})
    assert ckpt_name == 'epoch=4_val/loss=0.03000.ckpt'

    # TODO: Checks with filepath. To be removed in v1.2
    # CWD
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath='.').format_checkpoint_name(3, {})
    assert ckpt_name == str(Path('.').resolve() / 'epoch=3.ckpt')

    # dir does not exist so it is used as filename
    filepath = tmpdir / 'dir'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == tmpdir / 'test-dir.ckpt'

    # now, dir exists
    os.mkdir(filepath)
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath=filepath, prefix='test').format_checkpoint_name(3, {})
    assert ckpt_name == filepath / 'test-epoch=3.ckpt'


def test_model_checkpoint_save_last(tmpdir):
    """Tests that save_last produces only one last checkpoint."""
    model = EvalModelTemplate()
    epochs = 3
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last-{epoch}'
    model_checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
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
        ModelCheckpoint(dirpath=tmpdir, save_top_k=-3)


def test_none_monitor_top_k(tmpdir):
    """ Test that a warning appears for positive top_k with monitor=None. """
    with pytest.raises(
        MisconfigurationException, match=r'ModelCheckpoint\(save_top_k=3, monitor=None\) is not a valid*'
    ):
        ModelCheckpoint(dirpath=tmpdir, save_top_k=3)
    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, save_top_k=None)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=-1)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=0)


def test_none_monitor_save_last(tmpdir):
    """ Test that a warning appears for save_last=True with monitor=None. """
    with pytest.warns(
        UserWarning, match=r'ModelCheckpoint\(save_last=True, monitor=None\) is a redundant.*'
    ):
        ModelCheckpoint(dirpath=tmpdir, save_last=True)
    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, save_last=None)
    ModelCheckpoint(dirpath=tmpdir, save_last=False)


def test_model_checkpoint_none_monitor(tmpdir):
    """ Test that it is possible to save all checkpoints when monitor=None. """
    model = EvalModelTemplate()
    model.validation_step = model.validation_step_no_monitor
    model.validation_epoch_end = model.validation_epoch_end_no_monitor

    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmpdir, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
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
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, period=period)
    trainer = Trainer(
        default_root_dir=tmpdir,
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
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
    trainer = Trainer(
        default_root_dir=tmpdir,
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
    epochs = 3

    class CustomModel(EvalModelTemplate):
        def validation_epoch_end(self, outputs):
            return {'epoch': self.current_epoch}

    model = CustomModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="epoch", mode='max', save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)

    assert checkpoint_callback.monitor == 'epoch'
    assert checkpoint_callback.best_model_path == tmpdir / "epoch=2.ckpt"
    assert checkpoint_callback.best_model_score == epochs - 1
    assert len(os.listdir(tmpdir)) == len(checkpoint_callback.best_k_models) == epochs
    assert set(checkpoint_callback.best_k_models.keys()) == set(str(tmpdir / f"epoch={i}.ckpt") for i in range(epochs))
    assert checkpoint_callback.kth_best_model_path == tmpdir / 'epoch=0.ckpt'


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
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename="{val_loss:.2f}"),
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_default_checkpoint_behavior(tmpdir):
    seed_everything(1234)

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
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename="{val_loss:.2f}"),
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
        checkpoint_callback=ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir,
                                            save_top_k=0, save_last=save_last),
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
        monitor='early_stop_on', dirpath=tmpdir, save_top_k=num_epochs, save_last=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
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


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize('mode', ['min', 'max'])
def test_checkpointing_with_nan_as_first(tmpdir, mode):
    monitor = [float('nan')]
    monitor += [5, 7, 8] if mode == 'max' else [8, 7, 5]

    class CurrentModel(BoringModel):
        def validation_epoch_end(self, outputs):
            val_loss = monitor[self.current_epoch]
            self.log('abc', val_loss)

    model = CurrentModel()

    trainer = Trainer(
        checkpoint_callback=ModelCheckpoint(monitor='abc', mode=mode, save_top_k=1, dirpath=tmpdir),
        default_root_dir=tmpdir,
        val_check_interval=1.0,
        max_epochs=len(monitor),
    )
    trainer.fit(model)

    # check that last one is also the best one
    assert trainer.dev_debugger.checkpoint_callback_history[-1]['epoch'] == len(monitor) - 1


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_checkpoint_repeated_strategy(tmpdir):
    """
    This test validates that the checkpoint can be called when provided to callacks list
    """

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=tmpdir, filename="{epoch:02d}")

    class ExtendedBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"val_loss": loss}

    model = ExtendedBoringModel()
    model.validation_step_end = None
    model.validation_epoch_end = None
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
    assert os.listdir(tmpdir) == ['epoch=00.ckpt']

    def get_last_checkpoint():
        ckpts = os.listdir(tmpdir)
        ckpts_map = {int(x.split("=")[1].split('.')[0]): osp.join(tmpdir, x) for x in ckpts if "epoch" in x}
        num_ckpts = len(ckpts_map) - 1
        return ckpts_map[num_ckpts]

    for idx in range(1, 5):
        # load from checkpoint
        chk = get_last_checkpoint()
        model = BoringModel.load_from_checkpoint(chk)
        trainer = pl.Trainer(max_epochs=1,
                             limit_train_batches=2,
                             limit_val_batches=2,
                             limit_test_batches=2,
                             resume_from_checkpoint=chk)
        trainer.fit(model)
        trainer.test(model)

        assert str(os.listdir(tmpdir)) == "['epoch=00.ckpt']"


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_checkpoint_repeated_strategy_tmpdir(tmpdir):
    """
    This test validates that the checkpoint can be called when provided to callacks list
    """

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filepath=os.path.join(tmpdir, "{epoch:02d}"))

    class ExtendedBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"val_loss": loss}

    model = ExtendedBoringModel()
    model.validation_step_end = None
    model.validation_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        callbacks=[checkpoint_callback])

    trainer.fit(model)
    assert sorted(os.listdir(tmpdir)) == sorted(['epoch=00.ckpt', 'lightning_logs'])
    path_to_lightning_logs = osp.join(tmpdir, 'lightning_logs')
    assert sorted(os.listdir(path_to_lightning_logs)) == sorted(['version_0'])

    def get_last_checkpoint():
        ckpts = os.listdir(tmpdir)
        ckpts_map = {int(x.split("=")[1].split('.')[0]): osp.join(tmpdir, x) for x in ckpts if "epoch" in x}
        num_ckpts = len(ckpts_map) - 1
        return ckpts_map[num_ckpts]

    for idx in range(1, 5):

        # load from checkpoint
        chk = get_last_checkpoint()
        model = BoringModel.load_from_checkpoint(chk)
        trainer = pl.Trainer(default_root_dir=tmpdir,
                             max_epochs=1,
                             limit_train_batches=2,
                             limit_val_batches=2,
                             limit_test_batches=2,
                             resume_from_checkpoint=chk)

        trainer.fit(model)
        trainer.test(model)
        assert sorted(os.listdir(tmpdir)) == sorted(['epoch=00.ckpt', 'lightning_logs'])
        assert sorted(os.listdir(path_to_lightning_logs)) == sorted([f'version_{i}' for i in range(idx + 1)])


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_checkpoint_repeated_strategy_extended(tmpdir):
    """
    This test validates checkpoint can be called several times without
    increasing internally its global step if nothing run.
    """

    class ExtendedBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"val_loss": loss}

    def assert_trainer_init(trainer):
        assert not trainer.checkpoint_connector.has_trained
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def get_last_checkpoint(ckpt_dir):
        ckpts = os.listdir(ckpt_dir)
        ckpts.sort()
        return osp.join(ckpt_dir, ckpts[-1])

    def assert_checkpoint_content(ckpt_dir):
        chk = pl_load(get_last_checkpoint(ckpt_dir))
        assert chk["epoch"] == epochs
        assert chk["global_step"] == 4

    def assert_checkpoint_log_dir(idx):
        lightning_logs_path = osp.join(tmpdir, 'lightning_logs')
        assert sorted(os.listdir(lightning_logs_path)) == [f'version_{i}' for i in range(idx + 1)]
        assert len(os.listdir(ckpt_dir)) == epochs

    def get_model():
        model = ExtendedBoringModel()
        model.validation_step_end = None
        model.validation_epoch_end = None
        return model

    ckpt_dir = osp.join(tmpdir, 'checkpoints')
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    epochs = 2
    limit_train_batches = 2

    model = get_model()

    trainer_config = dict(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=3,
        limit_test_batches=4,
    )

    trainer = pl.Trainer(
        **trainer_config,
        checkpoint_callback=checkpoint_cb,
    )
    assert_trainer_init(trainer)

    trainer.fit(model)
    assert trainer.checkpoint_connector.has_trained
    assert trainer.global_step == epochs * limit_train_batches
    assert trainer.current_epoch == epochs - 1
    assert_checkpoint_log_dir(0)

    trainer.test(model)
    assert trainer.current_epoch == epochs - 1

    assert_checkpoint_content(ckpt_dir)

    for idx in range(1, 5):
        chk = get_last_checkpoint(ckpt_dir)
        assert_checkpoint_content(ckpt_dir)

        checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
        model = get_model()

        # load from checkpoint
        trainer = pl.Trainer(
            **trainer_config,
            resume_from_checkpoint=chk,
            checkpoint_callback=checkpoint_cb,
        )
        assert_trainer_init(trainer)

        trainer.test(model)
        assert not trainer.checkpoint_connector.has_trained
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs

        trainer.fit(model)
        assert not trainer.checkpoint_connector.has_trained
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert_checkpoint_log_dir(idx)


@pytest.mark.parametrize(
    'filepath, dirpath, filename',
    [
        (None, None, None),
        ('.', '.', None),
        ('', None, None),
        ('my/path/', 'my/', 'path'),
        ('my/path/{val_loss:.2f}', 'my/path/', '{val_loss:.2f}'),
    ]
)
def test_filepath_decomposition_dirpath_filename(tmpdir, filepath, dirpath, filename):
    mc_cb = ModelCheckpoint(filepath=filepath)
    dirpath = os.path.realpath(dirpath) if dirpath else dirpath

    assert mc_cb.dirpath == dirpath
    assert mc_cb.filename == filename


def test_configure_model_checkpoint(tmpdir):
    """ Test all valid and invalid ways a checkpoint callback can be passed to the Trainer. """
    kwargs = dict(default_root_dir=tmpdir)
    callback1 = ModelCheckpoint()
    callback2 = ModelCheckpoint()

    # no callbacks
    trainer = Trainer(checkpoint_callback=False, callbacks=[], **kwargs)
    assert not any(isinstance(c, ModelCheckpoint) for c in trainer.callbacks)
    assert trainer.checkpoint_callback is None

    # default configuration
    trainer = Trainer(checkpoint_callback=True, callbacks=[], **kwargs)
    assert len([c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]) == 1
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)

    # custom callback passed to callbacks list, checkpoint_callback=True is ignored
    trainer = Trainer(checkpoint_callback=True, callbacks=[callback1], **kwargs)
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
    assert trainer.checkpoint_callback == callback1

    # multiple checkpoint callbacks
    trainer = Trainer(callbacks=[callback1, callback2], **kwargs)
    assert trainer.checkpoint_callback == callback1
    assert trainer.checkpoint_callbacks == [callback1, callback2]

    with pytest.warns(DeprecationWarning, match='will no longer be supported in v1.3'):
        trainer = Trainer(checkpoint_callback=callback1, callbacks=[], **kwargs)
        assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
        assert trainer.checkpoint_callback == callback1

    with pytest.warns(DeprecationWarning, match="will no longer be supported in v1.3"):
        trainer = Trainer(checkpoint_callback=callback1, callbacks=[callback2], **kwargs)
        assert trainer.checkpoint_callback == callback2
        assert trainer.checkpoint_callbacks == [callback2, callback1]

    with pytest.raises(MisconfigurationException, match="checkpoint_callback=False but found ModelCheckpoint"):
        Trainer(checkpoint_callback=False, callbacks=[callback1], **kwargs)
