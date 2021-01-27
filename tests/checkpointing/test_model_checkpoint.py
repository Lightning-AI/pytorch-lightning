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
import logging
import os
import pickle
import platform
import re
from argparse import Namespace
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import cloudpickle
import pytest
import torch
import yaml
from omegaconf import Container, OmegaConf

import pytorch_lightning as pl
import tests.base.develop_utils as tutils
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel


class LogInTwoMethods(BoringModel):
    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        self.log('early_stop_on', out['loss'])
        return out

    def validation_epoch_end(self, outputs):
        outs = torch.stack([x['x'] for x in outputs]).mean()
        self.log('epoch', self.current_epoch, on_epoch=True)
        self.log('val_acc', outs, on_epoch=True)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize('save_top_k', [-1])
def test_model_checkpoint_correct_score(tmpdir, save_top_k):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path"""
    tutils.reset_seed()

    model = LogInTwoMethods()

    filename = "{val_acc:.4f}-{epoch}"

    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor='val_acc', save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=2)
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
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=None, filename='{epoch}', save_top_k=save_top_k)
    max_epochs = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint],
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
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)

    path_yaml = os.path.join(tmpdir, 'best_k_models.yaml')
    checkpoint.to_yaml(path_yaml)
    d = yaml.full_load(open(path_yaml, 'r'))
    best_k = {k: v for k, v in checkpoint.best_k_models.items()}
    assert d == best_k


@pytest.mark.parametrize(
    "logger_version,expected",
    [(None, "version_0"), (1, "version_1"), ("awesome", "awesome")],
)
def test_model_checkpoint_path(tmpdir, logger_version, expected):
    """Test that "version_" prefix is only added when logger's version is an integer"""
    tutils.reset_seed()
    model = LogInTwoMethods()
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
            # twice the calls expected because ddp broadcast also uses torch.save
            assert torch.save.call_count == self.expected_count * 2
        else:
            assert torch.save.call_count == 0


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Distributed training is not supported on Windows",
)
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = LogInTwoMethods()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor='early_stop_on', expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(
        accelerator="ddp_cpu",
        num_processes=2,
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
    )
    result = trainer.fit(model)
    assert 1 == result


def test_model_checkpoint_format_checkpoint_name(tmpdir):
    # empty filename:
    ckpt_name = ModelCheckpoint._format_checkpoint_name('', 3, 2, {})
    assert ckpt_name == 'epoch=3-step=2'

    ckpt_name = ModelCheckpoint._format_checkpoint_name(None, 3, 2, {}, prefix='test')
    assert ckpt_name == 'test-epoch=3-step=2'

    # no groups case:
    ckpt_name = ModelCheckpoint._format_checkpoint_name('ckpt', 3, 2, {}, prefix='test')
    assert ckpt_name == 'test-ckpt'

    # no prefix
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch:03d}-{acc}', 3, 2, {'acc': 0.03})
    assert ckpt_name == 'epoch=003-acc=0.03'

    # prefix
    char_org = ModelCheckpoint.CHECKPOINT_JOIN_CHAR
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = '@'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch},{acc:.5f}', 3, 2, {'acc': 0.03}, prefix='test')
    assert ckpt_name == 'test@epoch=3,acc=0.03000'
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = char_org

    # no dirpath set
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath=None).format_checkpoint_name(3, 2, {})
    assert ckpt_name == 'epoch=3-step=2.ckpt'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='').format_checkpoint_name(5, 4, {})
    assert ckpt_name == 'epoch=5-step=4.ckpt'

    # CWD
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='.').format_checkpoint_name(3, 4, {})
    assert ckpt_name == str(Path('.').resolve() / 'epoch=3-step=4.ckpt')

    # with ver
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', dirpath=tmpdir, filename='name', prefix='test'
    ).format_checkpoint_name(3, 2, {}, ver=3)
    assert ckpt_name == tmpdir / 'test-name-v3.ckpt'

    # using slashes
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', dirpath=None, filename='{epoch}_{val/loss:.5f}'
    ).format_checkpoint_name(4, 3, {'val/loss': 0.03})
    assert ckpt_name == 'epoch=4_val/loss=0.03000.ckpt'

    # TODO: Checks with filepath. To be removed in v1.2
    # CWD
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', filepath='.').format_checkpoint_name(3, 2, {})
    assert ckpt_name == str(Path('.').resolve() / 'epoch=3-step=2.ckpt')

    # dir does not exist so it is used as filename
    filepath = tmpdir / 'dir'
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', filepath=filepath, prefix='test'
    ).format_checkpoint_name(3, 2, {})
    assert ckpt_name == tmpdir / 'test-dir.ckpt'

    # now, dir exists
    os.mkdir(filepath)
    ckpt_name = ModelCheckpoint(
        monitor='early_stop_on', filepath=filepath, prefix='test'
    ).format_checkpoint_name(3, 2, {})
    assert ckpt_name == filepath / 'test-epoch=3-step=2.ckpt'


class ModelCheckpointExtensionTest(ModelCheckpoint):
    FILE_EXTENSION = '.tpkc'


def test_model_checkpoint_file_extension(tmpdir):
    """
    Test ModelCheckpoint with different file extension.
    """

    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpointExtensionTest(monitor='early_stop_on', dirpath=tmpdir, save_top_k=1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_steps=1,
        logger=False,
    )
    trainer.fit(model)

    expected = ['epoch=0-step=0.tpkc', 'last.tpkc']
    assert set(expected) == set(os.listdir(tmpdir))


def test_model_checkpoint_save_last(tmpdir):
    """Tests that save_last produces only one last checkpoint."""
    seed_everything()
    model = LogInTwoMethods()
    epochs = 3
    ModelCheckpoint.CHECKPOINT_NAME_LAST = 'last-{epoch}'
    model_checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=epochs,
        limit_train_batches=10,
        limit_val_batches=10,
        logger=False,
    )
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(
        ModelCheckpoint.CHECKPOINT_NAME_LAST, trainer.current_epoch, trainer.global_step, {}
    )
    last_filename = last_filename + '.ckpt'
    assert str(tmpdir / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmpdir)) == set(
        [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [9, 19, 29])] + [last_filename]
    )

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
    seed_everything()
    model = LogInTwoMethods()

    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmpdir, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)

    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == checkpoint_callback.last_model_path == tmpdir / 'epoch=1-step=19.ckpt'
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''

    # check that the correct ckpts were created
    expected = [f'epoch={i}-step={j}.ckpt' for i, j in zip(range(epochs), [9, 19])]
    assert set(os.listdir(tmpdir)) == set(expected)


@pytest.mark.parametrize("period", list(range(4)))
def test_model_checkpoint_period(tmpdir, period):
    model = LogInTwoMethods()
    epochs = 5
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}', save_top_k=-1, period=period)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        val_check_interval=1.0,
        logger=False,
    )
    trainer.fit(model)

    # check that the correct ckpts were created
    expected = [f'epoch={e}.ckpt' for e in range(epochs) if not (e + 1) % period] if period > 0 else []
    assert set(os.listdir(tmpdir)) == set(expected)


def test_model_checkpoint_topk_zero(tmpdir):
    """ Test that no checkpoints are saved when save_top_k=0. """
    model = LogInTwoMethods()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        max_epochs=2,
        logger=False,
    )
    trainer.fit(model)
    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == ''
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''
    # check that no ckpts were created
    assert len(os.listdir(tmpdir)) == 0


def test_model_checkpoint_topk_all(tmpdir):
    """ Test that save_top_k=-1 tracks the best models when monitor key is provided. """
    seed_everything(1000)
    epochs = 3

    class CustomModel(LogInTwoMethods):
        def validation_epoch_end(self, outputs):
            return {'epoch': self.current_epoch}

    model = CustomModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir,
        filename="{epoch}",
        monitor="epoch",
        mode='max',
        save_top_k=-1,
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        logger=False,
        val_check_interval=1.0,
    )
    trainer.fit(model)

    assert checkpoint_callback.monitor == 'epoch'
    assert checkpoint_callback.best_model_path == tmpdir / "epoch=2.ckpt"
    assert checkpoint_callback.best_model_score == epochs - 1
    assert len(os.listdir(tmpdir)) == len(checkpoint_callback.best_k_models) == epochs
    assert set(checkpoint_callback.best_k_models.keys()) == set(str(tmpdir / f"epoch={i}.ckpt") for i in range(epochs))
    assert checkpoint_callback.kth_best_model_path == tmpdir / 'epoch=0.ckpt'


def test_ckpt_metric_names(tmpdir):
    model = LogInTwoMethods()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        progress_bar_refresh_rate=0,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        callbacks=[ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename="{val_loss:.2f}")],
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
    os.environ['PL_DEV_DEBUG'] = '1'

    model = LogInTwoMethods()
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
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 3

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(os.path.join(tmpdir, 'lightning_logs', 'version_0', 'checkpoints'))
    assert len(ckpts) == 1
    assert ckpts[0] == 'epoch=2-step=14.ckpt'


def test_ckpt_metric_names_results(tmpdir):
    class ResultLog(BoringModel):
        def training_step(self, batch, batch_idx):
            y_hat = self(batch)

            # calculate loss
            loss_val = self.loss(batch, y_hat)
            log_val = loss_val

            # alternate between tensors and scalars for "log" and "progress_bar"
            if batch_idx % 2 == 0:
                log_val = log_val.item()

            result = pl.core.step_result.TrainResult(loss_val)
            result.log('some_val', log_val * log_val, prog_bar=True, logger=False)
            result.log('train_some_val', log_val * log_val)
            return result

        def validation_step(self, batch, batch_idx):
            y_hat = self(batch)

            loss_val = self.loss(batch, y_hat)

            # acc
            labels_hat = torch.argmax(y_hat, dim=1)
            val_acc = torch.sum(batch == labels_hat).item() / (len(batch) * 1.0)
            val_acc = torch.tensor(val_acc).type_as(batch)

            result = pl.core.step_result.EvalResult(checkpoint_on=loss_val, early_stop_on=loss_val)
            result.log_dict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return result

    model = ResultLog()
    model.training_step_end = None
    model.training_epoch_end = None
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
        callbacks=[ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename="{val_loss:.2f}")],
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
    model = LogInTwoMethods()
    if not should_validate:
        model.validation_step = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[ModelCheckpoint(monitor='early_stop_on', filepath=tmpdir,
                                   save_top_k=0, save_last=save_last)],
        max_epochs=max_epochs,
    )
    with caplog.at_level(logging.INFO):
        trainer.fit(model)
    assert caplog.messages.count('Saving latest checkpoint...') == save_last


def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    """ Tests that the save_last checkpoint contains the latest information. """
    seed_everything(100)
    model = LogInTwoMethods()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(
        monitor='early_stop_on', dirpath=tmpdir, filename="{epoch}", save_top_k=num_epochs, save_last=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
    )
    trainer.fit(model)

    path_last_epoch = str(tmpdir / f"epoch={num_epochs - 1}.ckpt")
    path_last = str(tmpdir / "last.ckpt")
    assert path_last == model_checkpoint.last_model_path
    assert os.path.isfile(path_last_epoch)

    ckpt_last_epoch = torch.load(path_last_epoch)
    ckpt_last = torch.load(path_last)
    assert all(ckpt_last_epoch[k] == ckpt_last[k] for k in ("epoch", "global_step"))

    ch_type = type(model_checkpoint)
    assert ckpt_last["callbacks"][ch_type] == ckpt_last_epoch["callbacks"][ch_type]

    # it is easier to load the model objects than to iterate over the raw dict of tensors
    model_last_epoch = LogInTwoMethods.load_from_checkpoint(path_last_epoch)
    model_last = LogInTwoMethods.load_from_checkpoint(
        model_checkpoint.last_model_path
    )
    for w0, w1 in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize('mode', ['min', 'max'])
def test_checkpointing_with_nan_as_first(tmpdir, mode):
    monitor = [float('nan')]
    monitor += [5, 7, 8] if mode == 'max' else [8, 7, 5]

    class CurrentModel(LogInTwoMethods):
        def validation_epoch_end(self, outputs):
            val_loss = monitor[self.current_epoch]
            self.log('abc', val_loss)

    model = CurrentModel()

    trainer = Trainer(
        callbacks=[ModelCheckpoint(monitor='abc', mode=mode, save_top_k=1, dirpath=tmpdir)],
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
    This test validates that the checkpoint can be called when provided to callbacks list
    """
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=tmpdir, filename="{epoch:02d}")

    class ExtendedBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"val_loss": loss}

    model = ExtendedBoringModel()
    model.validation_epoch_end = None
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        callbacks=[checkpoint_callback],
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)
    assert os.listdir(tmpdir) == ['epoch=00.ckpt']

    for idx in range(4):
        # load from checkpoint
        model = LogInTwoMethods.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer = pl.Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            resume_from_checkpoint=checkpoint_callback.best_model_path,
            weights_summary=None,
            progress_bar_refresh_rate=0,
        )
        trainer.fit(model)
        trainer.test(model, verbose=False)
    assert set(os.listdir(tmpdir)) == {'epoch=00.ckpt', 'lightning_logs'}
    assert set(os.listdir(tmpdir.join("lightning_logs"))) == {f'version_{i}' for i in range(4)}


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

        def validation_epoch_end(self, *_):
            ...

    def assert_trainer_init(trainer):
        assert not trainer.checkpoint_connector.has_trained
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def get_last_checkpoint(ckpt_dir):
        last = ckpt_dir.listdir(sort=True)[-1]
        return str(last)

    def assert_checkpoint_content(ckpt_dir):
        chk = pl_load(get_last_checkpoint(ckpt_dir))
        assert chk["epoch"] == epochs
        assert chk["global_step"] == 4

    def assert_checkpoint_log_dir(idx):
        lightning_logs = tmpdir / 'lightning_logs'
        actual = [d.basename for d in lightning_logs.listdir(sort=True)]
        assert actual == [f'version_{i}' for i in range(idx + 1)]
        assert len(ckpt_dir.listdir()) == epochs

    ckpt_dir = tmpdir / 'checkpoints'
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    epochs = 2
    limit_train_batches = 2
    trainer_config = dict(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=3,
        limit_test_batches=4,
        callbacks=[checkpoint_cb],
    )
    trainer = pl.Trainer(**trainer_config)
    assert_trainer_init(trainer)

    model = ExtendedBoringModel()
    trainer.fit(model)
    assert trainer.checkpoint_connector.has_trained
    assert trainer.global_step == epochs * limit_train_batches
    assert trainer.current_epoch == epochs - 1
    assert_checkpoint_log_dir(0)
    assert_checkpoint_content(ckpt_dir)

    trainer.test(model)
    assert trainer.current_epoch == epochs - 1

    for idx in range(1, 5):
        chk = get_last_checkpoint(ckpt_dir)
        assert_checkpoint_content(ckpt_dir)

        # load from checkpoint
        trainer_config["callbacks"] = [ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)]
        trainer = pl.Trainer(**trainer_config, resume_from_checkpoint=chk)
        assert_trainer_init(trainer)

        model = ExtendedBoringModel()
        trainer.test(model)
        assert not trainer.checkpoint_connector.has_trained
        # resume_from_checkpoint is resumed when calling `.fit`
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
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
        trainer = Trainer(checkpoint_callback=callback1, **kwargs)
        assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
        assert trainer.checkpoint_callback == callback1

    with pytest.warns(DeprecationWarning, match="will no longer be supported in v1.3"):
        trainer = Trainer(checkpoint_callback=callback1, callbacks=[callback2], **kwargs)
        assert trainer.checkpoint_callback == callback2
        assert trainer.checkpoint_callbacks == [callback2, callback1]

    with pytest.raises(MisconfigurationException, match="checkpoint_callback=False but found ModelCheckpoint"):
        Trainer(checkpoint_callback=False, callbacks=[callback1], **kwargs)


def test_val_check_interval_checkpoint_files(tmpdir):
    """ Test correct checkpoint naming when validating/checkpointing multiple times per epoch. """
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpoint(
        dirpath=tmpdir,
        save_top_k=-1,
        monitor="val_acc",
        mode="max",
        verbose=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        val_check_interval=0.2,
        max_epochs=1,
        limit_train_batches=10,
        callbacks=[model_checkpoint]
    )
    trainer.fit(model)
    files = sorted([p.name for p in Path(tmpdir).glob("*.ckpt")])
    assert files == [f"epoch=0-step={s}.ckpt" for s in [1, 3, 5, 7, 9]]


def test_current_score(tmpdir):
    """ Check that the current_score value is correct and was saved """
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", (self.current_epoch + 1) / 10)
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(
        dirpath=tmpdir,
        save_top_k=3,
        monitor="foo",
        mode="min",
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(TestModel())
    assert model_checkpoint.current_score == 0.3
    ckpts = [torch.load(str(ckpt)) for ckpt in tmpdir.listdir()]
    ckpts = [ckpt["callbacks"][type(model_checkpoint)] for ckpt in ckpts]
    assert sorted(ckpt["current_score"] for ckpt in ckpts) == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("mode", ["min", "max"])
def test_current_score_when_nan(tmpdir, mode):
    """ Check that ModelCheckpoint handles NaN values correctly """
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", float("nan"))
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(
        dirpath=tmpdir,
        save_top_k=1,
        monitor="foo",
        mode=mode,
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(TestModel())
    expected = float("inf" if mode == "min" else "-inf")
    assert model_checkpoint.best_model_score == expected
    assert model_checkpoint.current_score == expected


@pytest.mark.parametrize("hparams_type", [dict, Container])
def test_hparams_type(tmpdir, hparams_type):
    class TestModel(BoringModel):
        def __init__(self, hparams):
            super().__init__()
            self.save_hyperparameters(hparams)

    model_checkpoint = ModelCheckpoint(
        dirpath=tmpdir,
        save_top_k=1,
        monitor="foo",
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    hp = {"test_hp_0": 1, "test_hp_1": 2}
    hp = OmegaConf.create(hp) if hparams_type == Container else Namespace(**hp)
    model = TestModel(hp)
    trainer.fit(model)
    ckpt = trainer.checkpoint_connector.dump_checkpoint()
    if hparams_type == Container:
        assert isinstance(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY], hparams_type)
    else:
        # make sure it's not AttributeDict
        assert type(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY]) == hparams_type


@pytest.mark.parametrize('max_epochs', [3, 4])
@pytest.mark.parametrize(
    'save_top_k, expected',
    [
        (1, ['curr_epoch.ckpt']),
        (2, ['curr_epoch.ckpt', 'curr_epoch-v0.ckpt']),
    ]
)
def test_model_checkpoint_file_already_exists(tmpdir, max_epochs, save_top_k, expected):
    """
    Test that version is added to filename if required and it already exists in dirpath.
    """
    model_checkpoint = ModelCheckpoint(
        dirpath=tmpdir,
        filename='curr_epoch',
        save_top_k=save_top_k,
        monitor='epoch',
        mode='max',
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=max_epochs,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=None,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )

    model = BoringModel()
    trainer.fit(model)
    ckpt_files = os.listdir(tmpdir)
    assert set(ckpt_files) == set(expected)

    epochs_in_ckpt_files = [pl_load(os.path.join(tmpdir, f))['epoch'] - 1 for f in ckpt_files]
    assert sorted(epochs_in_ckpt_files) == list(range(max_epochs - save_top_k, max_epochs))
