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
import glob
import logging as log
import os
import pickle
from copy import deepcopy
from typing import Generic, TypeVar

import cloudpickle
import pytest
import torch
import torch.nn.functional as F

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import RunningStage, TrainerState
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel


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


class ValTestLossBoringModel(BoringModel):

    def __init__(self, batch_size=4):
        super().__init__()
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        self.log('val_loss', out['x'])
        return out

    def test_step(self, batch, batch_idx):
        out = super().test_step(batch, batch_idx)
        self.log('test_loss', out['y'])
        return out


T = TypeVar('T')


class GenericParentValTestLossBoringModel(Generic[T], ValTestLossBoringModel):

    def __init__(self, batch_size: int = 4):
        super().__init__(batch_size=batch_size)


class GenericValTestLossBoringModel(GenericParentValTestLossBoringModel[int]):
    pass


def test_model_properties_resume_from_checkpoint(tmpdir):
    """
    Test that properties like `current_epoch` and `global_step`
    in model and trainer are always the same.
    """
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", save_last=True)
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
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
    dm = ClassifDataModule()
    model = ClassificationModel()
    checkpoint_cb = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=False,
        callbacks=[checkpoint_cb],
        limit_train_batches=2,
        limit_val_batches=2,
    )
    # Generate checkpoint `last.ckpt` with BoringModel
    trainer.fit(model, datamodule=dm)
    # `True` if resume/restore successfully else `False`
    assert trainer.checkpoint_connector.restore(str(tmpdir / "last.ckpt"), trainer.on_gpu)
    assert not trainer.checkpoint_connector.restore(str(tmpdir / "last_non_existing.ckpt"), trainer.on_gpu)


class CaptureCallbacksBeforeTraining(Callback):
    callbacks = []

    def on_train_start(self, trainer, pl_module):
        self.callbacks = deepcopy(trainer.callbacks)


def test_callbacks_state_resume_from_checkpoint(tmpdir):
    """ Test that resuming from a checkpoint restores callbacks that persist state. """
    dm = ClassifDataModule()
    model = ClassificationModel()
    callback_capture = CaptureCallbacksBeforeTraining()

    def get_trainer_args():
        checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", save_last=True)
        trainer_args = dict(
            default_root_dir=tmpdir, max_steps=1, logger=False, callbacks=[
                checkpoint,
                callback_capture,
            ]
        )
        assert checkpoint.best_model_path == ""
        assert checkpoint.best_model_score is None
        return trainer_args

    # initial training
    trainer = Trainer(**get_trainer_args())
    trainer.fit(model, datamodule=dm)
    callbacks_before_resume = deepcopy(trainer.callbacks)

    # resumed training
    trainer = Trainer(**get_trainer_args(), resume_from_checkpoint=str(tmpdir / "last.ckpt"))
    trainer.fit(model, datamodule=dm)

    assert len(callbacks_before_resume) == len(callback_capture.callbacks)

    for before, after in zip(callbacks_before_resume, callback_capture.callbacks):
        if isinstance(before, ModelCheckpoint):
            assert before.best_model_path == after.best_model_path
            assert before.best_model_score == after.best_model_score


def test_callbacks_references_resume_from_checkpoint(tmpdir):
    """ Test that resuming from a checkpoint sets references as expected. """
    dm = ClassifDataModule()
    model = ClassificationModel()
    args = {'default_root_dir': tmpdir, 'max_steps': 1, 'logger': False}

    # initial training
    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", save_last=True)
    trainer = Trainer(**args, callbacks=[checkpoint])
    assert checkpoint is trainer.callbacks[-1] is trainer.checkpoint_callback
    trainer.fit(model, datamodule=dm)

    # resumed training
    new_checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", save_last=True)
    # pass in a new checkpoint object, which should take
    # precedence over the one in the last.ckpt file
    trainer = Trainer(**args, callbacks=[new_checkpoint], resume_from_checkpoint=str(tmpdir / "last.ckpt"))
    assert checkpoint is not new_checkpoint
    assert new_checkpoint is trainer.callbacks[-1] is trainer.checkpoint_callback
    trainer.fit(model, datamodule=dm)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_running_test_pretrained_model_distrib_dp(tmpdir):
    """Verify `test()` on pretrained model."""

    tutils.set_random_master_port()

    class CustomClassificationModelDP(ClassificationModel):

        def _step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            return {'logits': logits, 'y': y}

        def training_step(self, batch, batch_idx):
            _, y = batch
            out = self._step(batch, batch_idx)
            loss = F.cross_entropy(out['logits'], y)
            return loss

        def validation_step(self, batch, batch_idx):
            return self._step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            return self._step(batch, batch_idx)

        def validation_step_end(self, outputs):
            self.log('val_acc', self.valid_acc(outputs['logits'], outputs['y']))

    dm = ClassifDataModule()
    model = CustomClassificationModelDP(lr=0.1)

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=5,
        callbacks=[checkpoint],
        logger=logger,
        gpus=[0, 1],
        accelerator='dp',
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=dm)

    # correct result and ok accuracy
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    pretrained_model = ClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)
    pretrained_model.cpu()

    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tpipes.run_prediction(pretrained_model, dataloader)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_running_test_pretrained_model_distrib_ddp_spawn(tmpdir):
    """Verify `test()` on pretrained model."""
    tutils.set_random_master_port()
    dm = ClassifDataModule()
    model = ClassificationModel()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[checkpoint],
        logger=logger,
        gpus=[0, 1],
        accelerator='ddp_spawn',
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=dm)

    log.info(os.listdir(tutils.get_data_path(logger, path_dir=tmpdir)))

    # correct result and ok accuracy
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    pretrained_model = ClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)
    pretrained_model.cpu()

    dataloaders = dm.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tpipes.run_prediction(pretrained_model, dataloader, min_acc=0.1)


def test_running_test_pretrained_model_cpu(tmpdir):
    """Verify test() on pretrained model."""
    tutils.reset_seed()
    dm = ClassifDataModule()
    model = ClassificationModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        callbacks=[checkpoint],
        logger=logger,
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=dm)

    # correct result and ok accuracy
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    pretrained_model = ClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model, datamodule=dm)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer, key='test_acc', thr=0.45)


@pytest.mark.parametrize('model_template', [ValTestLossBoringModel, GenericValTestLossBoringModel])
def test_load_model_from_checkpoint(tmpdir, model_template):
    """Verify test() on pretrained model."""
    tutils.reset_seed()
    model = model_template()

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='val_loss', save_top_k=-1)],
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)
    trainer.test(ckpt_path=None)

    # correct result and ok accuracy
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    # load last checkpoint
    last_checkpoint = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt")))[-1]

    # Since `BoringModel` has `_save_hparams = True` by default, check that ckpt has hparams
    ckpt = torch.load(last_checkpoint)
    assert model_template.CHECKPOINT_HYPER_PARAMS_KEY in ckpt.keys(), 'hyper_parameters missing from checkpoints'

    # Ensure that model can be correctly restored from checkpoint
    pretrained_model = model_template.load_from_checkpoint(last_checkpoint)

    # test that hparams loaded correctly
    for k, v in model.hparams.items():
        assert getattr(pretrained_model.hparams, k) == v

    # assert weights are the same
    for (old_name, old_p), (new_name, new_p) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        assert torch.all(torch.eq(old_p, new_p)), 'loaded weights are not the same as the saved weights'

    # Check `test` on pretrained model:
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_dp_resume(tmpdir):
    """Make sure DP continues training correctly."""
    model = BoringModel()

    trainer_options = dict(max_epochs=1, gpus=2, accelerator='dp', default_root_dir=tmpdir)

    # get logger
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['logger'] = logger
    trainer_options['callbacks'] = [checkpoint]

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.is_slurm_managing_tasks = True
    trainer.fit(model)

    # track epoch before saving. Increment since we finished the current epoch, don't want to rerun
    real_global_epoch = trainer.current_epoch + 1

    # correct result and ok accuracy
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    # ---------------------------
    # HPC LOAD/SAVE
    # ---------------------------
    # save
    trainer.checkpoint_connector.hpc_save(tmpdir, logger)

    # init new trainer
    new_logger = tutils.get_default_logger(tmpdir, version=logger.version)
    trainer_options['logger'] = new_logger
    trainer_options['callbacks'] = [ModelCheckpoint(dirpath=tmpdir)]
    trainer_options['limit_train_batches'] = 0.5
    trainer_options['limit_val_batches'] = 0.2
    trainer_options['max_epochs'] = 1
    new_trainer = Trainer(**trainer_options)

    class CustomModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.on_train_start_called = False

        # set the epoch start hook so we can predict before the model does the full training
        def on_train_start(self):
            assert self.trainer.current_epoch == real_global_epoch and self.trainer.current_epoch > 0

            # if model and state loaded correctly, predictions will be good even though we
            # haven't trained with the new loaded model
            dp_model = new_trainer.model
            dp_model.eval()
            dp_model.module.module.running_stage = RunningStage.EVALUATING

            dataloader = self.train_dataloader()
            tpipes.run_prediction(self.trainer.lightning_module, dataloader)
            self.on_train_start_called = True

    # new model
    model = CustomModel()

    # fit new model which should load hpc weights
    new_trainer.fit(model)
    assert model.on_train_start_called

    # test freeze on gpu
    model.freeze()
    model.unfreeze()


def test_model_saving_loading(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
        default_root_dir=tmpdir,
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    batch = next(iter(dataloaders[0]))

    # generate preds before saving model
    model.eval()
    pred_before_saving = model(batch)

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    model_2 = BoringModel.load_from_checkpoint(
        checkpoint_path=new_weights_path,
        hparams_file=hparams_path,
    )
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(batch)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_strict_model_load_more_params(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = BoringModel()
    # Extra layer
    model.c_d3 = torch.nn.Linear(32, 32)

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmpdir), 'hparams.yaml')
    hparams_url = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}'
    ckpt_path = hparams_url if url_ckpt else new_weights_path

    BoringModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
        strict=False,
    )

    with pytest.raises(RuntimeError, match=r'Unexpected key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        BoringModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
            strict=True,
        )


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_strict_model_load_less_params(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmpdir), 'hparams.yaml')
    hparams_url = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}'
    ckpt_path = hparams_url if url_ckpt else new_weights_path

    class CurrentModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.c_d3 = torch.nn.Linear(7, 7)

    CurrentModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
        strict=False,
    )

    with pytest.raises(RuntimeError, match=r'Missing key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        CurrentModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
            strict=True,
        )


def test_model_pickle(tmpdir):
    model = BoringModel()
    pickle.dumps(model)
    cloudpickle.dumps(model)
