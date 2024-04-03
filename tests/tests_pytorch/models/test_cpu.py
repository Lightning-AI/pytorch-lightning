# Copyright The Lightning AI team.
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
from unittest import mock

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel

import tests_pytorch.helpers.pipelines as tpipes
import tests_pytorch.helpers.utils as tutils
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@mock.patch("lightning.fabric.plugins.environments.slurm.SLURMEnvironment.detect", return_value=True)
def test_cpu_slurm_save_load(_, tmp_path):
    """Verify model save/load/checkpoint on CPU."""
    seed_everything(42)

    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)
    version = logger.version

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        logger=logger,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
    )
    trainer.fit(model)
    real_global_step = trainer.global_step

    # traning complete
    assert trainer.state.finished, "cpu model failed to complete"

    # predict with trained model before saving
    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    model.eval()
    pred_before_saving = model(batch)

    # test HPC saving
    # simulate snapshot on slurm
    # save logger to make sure we get all the metrics
    if logger:
        logger.finalize("finished")
    hpc_save_path = trainer._checkpoint_connector.hpc_save_path(trainer.default_root_dir)
    trainer.save_checkpoint(hpc_save_path)
    assert os.path.exists(hpc_save_path)

    # new logger file to get meta
    logger = tutils.get_default_logger(tmp_path, version=version)

    model = BoringModel()

    class _StartCallback(Callback):
        # set the epoch start hook so we can predict before the model does the full training
        def on_train_epoch_start(self, trainer, model):
            assert trainer.global_step == real_global_step
            assert trainer.global_step > 0
            # predict with loaded model to make sure answers are the same
            mode = model.training
            model.eval()
            new_pred = model(batch)
            assert torch.eq(pred_before_saving, new_pred).all()
            model.train(mode)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        logger=logger,
        callbacks=[_StartCallback(), ModelCheckpoint(dirpath=tmp_path)],
    )
    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)


def test_early_stopping_cpu_model(tmp_path):
    seed_everything(42)

    class ModelTrainVal(BoringModel):
        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log("val_loss", output["x"])
            return output

    stopping = EarlyStopping(monitor="val_loss", min_delta=0.1)
    trainer_options = {
        "callbacks": [stopping],
        "default_root_dir": tmp_path,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "accumulate_grad_batches": 2,
        "limit_train_batches": 0.3,
        "limit_val_batches": 0.1,
    }

    model = ModelTrainVal()
    tpipes.run_model_test(trainer_options, model)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


@RunIf(skip_windows=True, sklearn=True)
def test_multi_cpu_model_ddp(tmp_path):
    """Make sure DDP works."""
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "cpu",
        "devices": 2,
        "strategy": "ddp_spawn",
    }

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, data=dm)


def test_lbfgs_cpu_model(tmp_path):
    """Test each of the trainer options.

    Testing LBFGS optimizer

    """
    seed_everything(42)

    class ModelSpecifiedOptimizer(BoringModel):
        def __init__(self, optimizer_name, learning_rate):
            super().__init__()
            self.optimizer_name = optimizer_name
            self.learning_rate = learning_rate
            self.save_hyperparameters()

    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "enable_progress_bar": False,
        "limit_train_batches": 0.2,
        "limit_val_batches": 0.2,
    }

    model = ModelSpecifiedOptimizer(optimizer_name="LBFGS", learning_rate=0.004)
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)


def test_default_logger_callbacks_cpu_model(tmp_path):
    """Test each of the trainer options."""
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "gradient_clip_val": 1.0,
        "overfit_batches": 0.20,
        "enable_progress_bar": False,
        "limit_train_batches": 0.01,
        "limit_val_batches": 0.01,
    }

    model = BoringModel()
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_running_test_after_fitting(tmp_path):
    """Verify test() on fitted model."""
    seed_everything(42)

    class ModelTrainValTest(BoringModel):
        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log("val_loss", output["x"])
            return output

        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log("test_loss", output["y"])
            return output

    model = ModelTrainValTest()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key="test_loss", thr=0.5)


def test_running_test_no_val(tmp_path):
    """Verify `test()` works on a model with no `val_dataloader`.

    It performs train and test only

    """
    seed_everything(42)

    class ModelTrainTest(BoringModel):
        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log("test_loss", output["y"])
            return output

        val_dataloader = None

    model = ModelTrainTest()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key="test_loss")


def test_cpu_model(tmp_path):
    """Make sure model trains on CPU."""
    seed_everything(42)
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 4,
        "limit_val_batches": 4,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


def test_all_features_cpu_model(tmp_path):
    """Test each of the trainer options."""
    seed_everything(42)
    trainer_options = {
        "default_root_dir": tmp_path,
        "gradient_clip_val": 1.0,
        "overfit_batches": 0.20,
        "enable_progress_bar": False,
        "accumulate_grad_batches": 2,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.4,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)
