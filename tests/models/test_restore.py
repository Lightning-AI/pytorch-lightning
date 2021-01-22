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
from copy import deepcopy
import glob
import logging as log
import os
import pickle

import cloudpickle
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from tests.base import BoringModel, EvalModelTemplate, GenericEvalModelTemplate, TrialMNIST


class ModelTrainerPropertyParity(Callback):

    def _check_properties(self, trainer, pl_module):
        assert trainer.global_step == pl_module.global_step
        assert trainer.current_epoch == pl_module.current_epoch

    def on_train_start(self, trainer, pl_module):
        self._check_properties(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        self._check_properties(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self._check_properties(trainer, pl_module)

    def on_epoch_end(self, trainer, pl_module):
        self._check_properties(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._check_properties(trainer, pl_module)


def test_model_properties_resume_from_checkpoint(tmpdir):
    """ Test that properties like `current_epoch` and `global_step`
    in model and trainer are always the same. """
    model = EvalModelTemplate()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_last=True)
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=False,
        callbacks=[checkpoint_callback, ModelTrainerPropertyParity()],  # this performs the assertions
    )
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    trainer_args.update(max_epochs=2)
    trainer = Trainer(**trainer_args, resume_from_checkpoint=str(tmpdir / "last.ckpt"))
    trainer.fit(model)


def test_try_resume_from_non_existing_checkpoint(tmpdir):
    """ Test that trying to resume from non-existing `resume_from_checkpoint` fail without error."""
    model = BoringModel()
    checkpoint_cb = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=False,
        callbacks=[checkpoint_cb],
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )
    # Generate checkpoint `last.ckpt` with BoringModel
    trainer.fit(model)
    # `True` if resume/restore successfully else `False`
    assert trainer.checkpoint_connector.restore(str(tmpdir / "last.ckpt"), trainer.on_gpu)
    assert not trainer.checkpoint_connector.restore(str(tmpdir / "last_non_existing.ckpt"), trainer.on_gpu)


class CaptureCallbacksBeforeTraining(Callback):
    callbacks = []

    def on_train_start(self, trainer, pl_module):
        self.callbacks = deepcopy(trainer.callbacks)


def test_callbacks_state_resume_from_checkpoint(tmpdir):
    """ Test that resuming from a checkpoint restores callbacks that persist state. """
    model = EvalModelTemplate()
    callback_capture = CaptureCallbacksBeforeTraining()

    def get_trainer_args():
        checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_last=True)
        trainer_args = dict(
            default_root_dir=tmpdir,
            max_steps=1,
            logger=False,
            callbacks=[
                checkpoint,
                callback_capture,
            ]
        )
        assert checkpoint.best_model_path == ""
        assert checkpoint.best_model_score is None
        return trainer_args

    # initial training
    trainer = Trainer(**get_trainer_args())
    trainer.fit(model)
    callbacks_before_resume = deepcopy(trainer.callbacks)

    # resumed training
    trainer = Trainer(**get_trainer_args(), resume_from_checkpoint=str(tmpdir / "last.ckpt"))
    trainer.fit(model)

    assert len(callbacks_before_resume) == len(callback_capture.callbacks)

    for before, after in zip(callbacks_before_resume, callback_capture.callbacks):
        if isinstance(before, ModelCheckpoint):
            assert before.best_model_path == after.best_model_path
            assert before.best_model_score == after.best_model_score


def test_callbacks_references_resume_from_checkpoint(tmpdir):
    """ Test that resuming from a checkpoint sets references as expected. """
    model = EvalModelTemplate()
    args = {'default_root_dir': tmpdir, 'max_steps': 1, 'logger': False}

    # initial training
    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_last=True)
    trainer = Trainer(**args, callbacks=[checkpoint])
    assert checkpoint is trainer.callbacks[0] is trainer.checkpoint_callback
    trainer.fit(model)

    # resumed training
    new_checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_last=True)
    # pass in a new checkpoint object, which should take
    # precedence over the one in the last.ckpt file
    trainer = Trainer(**args, callbacks=[new_checkpoint], resume_from_checkpoint=str(tmpdir / "last.ckpt"))
    assert checkpoint is not new_checkpoint
    assert new_checkpoint is trainer.callbacks[0] is trainer.checkpoint_callback
    trainer.fit(model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_running_test_pretrained_model_distrib_dp(tmpdir):
    """Verify `test()` on pretrained model."""
    tutils.set_random_master_port()

    model = EvalModelTemplate()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
        gpus=[0, 1],
        accelerator='dp',
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = EvalModelTemplate.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # run test set
    new_trainer = Trainer(**trainer_options)
    results = new_trainer.test(pretrained_model)
    pretrained_model.cpu()

    # test we have good test accuracy
    acc = results[0]['test_acc']
    assert acc > 0.5, f"Model failed to get expected {0.5} accuracy. test_acc = {acc}"

    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tpipes.run_prediction(dataloader, pretrained_model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_running_test_pretrained_model_distrib_ddp_spawn(tmpdir):
    """Verify `test()` on pretrained model."""
    tutils.set_random_master_port()

    model = EvalModelTemplate()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
        gpus=[0, 1],
        accelerator='ddp_spawn',
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    log.info(os.listdir(tutils.get_data_path(logger, path_dir=tmpdir)))

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = EvalModelTemplate.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # run test set
    new_trainer = Trainer(**trainer_options)
    results = new_trainer.test(pretrained_model)
    pretrained_model.cpu()

    acc = results[0]['test_acc']
    assert acc > 0.5, f"Model failed to get expected {0.5} accuracy. test_acc = {acc}"

    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tpipes.run_prediction(dataloader, pretrained_model)


def test_running_test_pretrained_model_cpu(tmpdir):
    """Verify test() on pretrained model."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=3,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = EvalModelTemplate.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer)


@pytest.mark.parametrize('model_template', [EvalModelTemplate, GenericEvalModelTemplate])
def test_load_model_from_checkpoint(tmpdir, model_template):
    """Verify test() on pretrained model."""
    hparams = model_template.get_default_hparams()
    model = model_template(**hparams)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_top_k=-1)],
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    trainer.test(ckpt_path=None)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'

    # load last checkpoint
    last_checkpoint = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt")))[-1]

    # Since `EvalModelTemplate` has `_save_hparams = True` by default, check that ckpt has hparams
    ckpt = torch.load(last_checkpoint)
    assert model_template.CHECKPOINT_HYPER_PARAMS_KEY in ckpt.keys(), 'hyper_parameters missing from checkpoints'

    # Ensure that model can be correctly restored from checkpoint
    pretrained_model = model_template.load_from_checkpoint(last_checkpoint)

    # test that hparams loaded correctly
    for k, v in hparams.items():
        assert getattr(pretrained_model, k) == v

    # assert weights are the same
    for (old_name, old_p), (new_name, new_p) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        assert torch.all(torch.eq(old_p, new_p)), 'loaded weights are not the same as the saved weights'

    # Check `test` on pretrained model:
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_dp_resume(tmpdir):
    """Make sure DP continues training correctly."""
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer_options = dict(max_epochs=1, gpus=2, accelerator='dp', default_root_dir=tmpdir)

    # get logger
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['logger'] = logger
    trainer_options['checkpoint_callback'] = checkpoint

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.is_slurm_managing_tasks = True
    result = trainer.fit(model)

    # track epoch before saving. Increment since we finished the current epoch, don't want to rerun
    real_global_epoch = trainer.current_epoch + 1

    # correct result and ok accuracy
    assert result == 1, 'amp + dp model failed to complete'

    # ---------------------------
    # HPC LOAD/SAVE
    # ---------------------------
    # save
    trainer.checkpoint_connector.hpc_save(tmpdir, logger)

    # init new trainer
    new_logger = tutils.get_default_logger(tmpdir, version=logger.version)
    trainer_options['logger'] = new_logger
    trainer_options['checkpoint_callback'] = ModelCheckpoint(dirpath=tmpdir)
    trainer_options['limit_train_batches'] = 0.5
    trainer_options['limit_val_batches'] = 0.2
    trainer_options['max_epochs'] = 1
    new_trainer = Trainer(**trainer_options)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_good_acc():
        assert new_trainer.current_epoch == real_global_epoch and new_trainer.current_epoch > 0

        # if model and state loaded correctly, predictions will be good even though we
        # haven't trained with the new loaded model
        dp_model = new_trainer.model
        dp_model.eval()

        dataloader = trainer.train_dataloader
        tpipes.run_prediction(dataloader, dp_model, dp=True)

    # new model
    model = EvalModelTemplate(**hparams)
    model.on_train_start = assert_good_acc

    # fit new model which should load hpc weights
    new_trainer.fit(model)

    # test freeze on gpu
    model.freeze()
    model.unfreeze()


def test_model_saving_loading(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
        default_root_dir=tmpdir,
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    x, y = batch
    x = x.view(x.size(0), -1)

    # generate preds before saving model
    model.eval()
    pred_before_saving = model(x)

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    model_2 = EvalModelTemplate.load_from_checkpoint(checkpoint_path=new_weights_path, hparams_file=hparams_path,)
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(x)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_strict_model_load_more_params(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = EvalModelTemplate()
    # Extra layer
    model.c_d3 = torch.nn.Linear(model.hidden_dim, model.hidden_dim)

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmpdir), 'hparams.yaml')
    hparams_url = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}'
    ckpt_path = hparams_url if url_ckpt else new_weights_path

    EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=False,
    )

    with pytest.raises(RuntimeError, match=r'Unexpected key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=True,
        )


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_strict_model_load_less_params(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmpdir), 'hparams.yaml')
    hparams_url = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}'
    ckpt_path = hparams_url if url_ckpt else new_weights_path

    class CurrentModel(EvalModelTemplate):
        def __init__(self):
            super().__init__()
            self.c_d3 = torch.nn.Linear(7, 7)

    CurrentModel.load_from_checkpoint(
        checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=False,
    )

    with pytest.raises(RuntimeError, match=r'Missing key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        CurrentModel.load_from_checkpoint(
            checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=True,
        )


def test_model_pickle(tmpdir):
    model = EvalModelTemplate()
    pickle.dumps(model)
    cloudpickle.dumps(model)
