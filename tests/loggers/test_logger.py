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
import pickle
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger, LoggerCollection, TensorBoardLogger
from pytorch_lightning.loggers.logger import DummyExperiment, DummyLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.logger import _convert_params, _sanitize_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tests.helpers.boring_model import BoringDataModule, BoringModel


def test_logger_collection():
    mock1 = MagicMock()
    mock2 = MagicMock()

    logger = LoggerCollection([mock1, mock2])

    assert logger[0] == mock1
    assert logger[1] == mock2

    assert logger.experiment[0] == mock1.experiment
    assert logger.experiment[1] == mock2.experiment

    assert logger.save_dir is None

    logger.update_agg_funcs({"test": np.mean}, np.sum)
    mock1.update_agg_funcs.assert_called_once_with({"test": np.mean}, np.sum)
    mock2.update_agg_funcs.assert_called_once_with({"test": np.mean}, np.sum)

    logger.log_metrics(metrics={"test": 2.0}, step=4)
    mock1.log_metrics.assert_called_once_with(metrics={"test": 2.0}, step=4)
    mock2.log_metrics.assert_called_once_with(metrics={"test": 2.0}, step=4)

    logger.finalize("success")
    mock1.finalize.assert_called_once()
    mock2.finalize.assert_called_once()


def test_logger_collection_unique_names():
    unique_name = "name1"
    logger1 = CustomLogger(name=unique_name)
    logger2 = CustomLogger(name=unique_name)

    logger = LoggerCollection([logger1, logger2])

    assert logger.name == unique_name


def test_logger_collection_names_order():
    loggers = [CustomLogger(name=n) for n in ("name1", "name2", "name1", "name3")]
    logger = LoggerCollection(loggers)
    assert logger.name == f"{loggers[0].name}_{loggers[1].name}_{loggers[3].name}"


def test_logger_collection_unique_versions():
    unique_version = "1"
    logger1 = CustomLogger(version=unique_version)
    logger2 = CustomLogger(version=unique_version)

    logger = LoggerCollection([logger1, logger2])

    assert logger.version == unique_version


def test_logger_collection_versions_order():
    loggers = [CustomLogger(version=v) for v in ("1", "2", "1", "3")]
    logger = LoggerCollection(loggers)
    assert logger.version == f"{loggers[0].version}_{loggers[1].version}_{loggers[3].version}"


class CustomLogger(Logger):
    def __init__(self, experiment: str = "test", name: str = "name", version: str = "1"):
        super().__init__()
        self._experiment = experiment
        self._name = name
        self._version = version
        self.hparams_logged = None
        self.metrics_logged = {}
        self.finalized = False
        self.after_save_checkpoint_called = False

    @property
    def experiment(self):
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams_logged = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.metrics_logged = metrics

    @rank_zero_only
    def finalize(self, status):
        self.finalized_status = status

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def after_save_checkpoint(self, checkpoint_callback):
        self.after_save_checkpoint_called = True


def test_custom_logger(tmpdir):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("train_loss", loss)
            return {"loss": loss}

    logger = CustomLogger()
    model = CustomModel()
    trainer = Trainer(max_steps=2, log_every_n_steps=1, logger=logger, default_root_dir=tmpdir)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert logger.metrics_logged != {}
    assert logger.after_save_checkpoint_called
    assert logger.finalized_status == "success"


def test_multiple_loggers(tmpdir):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("train_loss", loss)
            return {"loss": loss}

    model = CustomModel()
    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(max_steps=2, log_every_n_steps=1, logger=[logger1, logger2], default_root_dir=tmpdir)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert logger1.hparams_logged is None
    assert logger1.metrics_logged != {}
    assert logger1.finalized_status == "success"

    assert logger2.hparams_logged is None
    assert logger2.metrics_logged != {}
    assert logger2.finalized_status == "success"


def test_multiple_loggers_pickle(tmpdir):
    """Verify that pickling trainer with multiple loggers works."""

    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(logger=[logger1, logger2])
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0}, 0)

    assert trainer2.logger[0].metrics_logged == {"acc": 1.0}
    assert trainer2.logger[1].metrics_logged == {"acc": 1.0}


def test_adding_step_key(tmpdir):
    class CustomTensorBoardLogger(TensorBoardLogger):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.logged_step = 0

        def log_metrics(self, metrics, step):
            if "val_acc" in metrics:
                assert step == self.logged_step

            super().log_metrics(metrics, step)

    class CustomModel(BoringModel):
        def training_epoch_end(self, outputs):
            self.logger.logged_step += 1
            self.log_dict({"step": self.logger.logged_step, "train_acc": self.logger.logged_step / 10})

        def validation_epoch_end(self, outputs):
            self.logger.logged_step += 1
            self.log_dict({"step": self.logger.logged_step, "val_acc": self.logger.logged_step / 10})

    model = CustomModel()
    trainer = Trainer(
        max_epochs=3,
        logger=CustomTensorBoardLogger(save_dir=tmpdir),
        default_root_dir=tmpdir,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)


def test_dummyexperiment_support_indexing():
    """Test that the DummyExperiment can imitate indexing the experiment in a LoggerCollection."""
    experiment = DummyExperiment()
    assert experiment[0] == experiment


def test_dummylogger_support_indexing():
    """Test that the DummyLogger can imitate indexing of a LoggerCollection."""
    logger = DummyLogger()
    assert logger[0] == logger


def test_dummylogger_empty_iterable():
    """Test that DummyLogger represents an empty iterable."""
    logger = DummyLogger()
    for _ in logger:
        assert False


def test_dummylogger_noop_method_calls():
    """Test that the DummyLogger methods can be called with arbitrary arguments."""
    logger = DummyLogger()
    logger.log_hyperparams("1", 2, three="three")
    logger.log_metrics("1", 2, three="three")


def test_dummyexperiment_support_item_assignment():
    """Test that the DummyExperiment supports item assignment."""
    experiment = DummyExperiment()
    experiment["variable"] = "value"
    assert experiment["variable"] != "value"  # this is only a stateless mock experiment


def test_np_sanitization():
    class CustomParamsLogger(CustomLogger):
        def __init__(self):
            super().__init__()
            self.logged_params = None

        @rank_zero_only
        def log_hyperparams(self, params):
            params = _convert_params(params)
            params = _sanitize_params(params)
            self.logged_params = params

    logger = CustomParamsLogger()
    np_params = {
        "np.bool_": np.bool_(1),
        "np.byte": np.byte(2),
        "np.intc": np.intc(3),
        "np.int_": np.int_(4),
        "np.longlong": np.longlong(5),
        "np.single": np.single(6.0),
        "np.double": np.double(8.9),
        "np.csingle": np.csingle(7 + 2j),
        "np.cdouble": np.cdouble(9 + 4j),
    }
    sanitized_params = {
        "np.bool_": True,
        "np.byte": 2,
        "np.intc": 3,
        "np.int_": 4,
        "np.longlong": 5,
        "np.single": 6.0,
        "np.double": 8.9,
        "np.csingle": "(7+2j)",
        "np.cdouble": "(9+4j)",
    }
    logger.log_hyperparams(Namespace(**np_params))
    assert logger.logged_params == sanitized_params


@pytest.mark.parametrize("logger", [True, False])
@patch("pytorch_lightning.loggers.tensorboard.TensorBoardLogger.log_hyperparams")
def test_log_hyperparams_being_called(log_hyperparams_mock, tmpdir, logger):
    class TestModel(BoringModel):
        def __init__(self, param_one, param_two):
            super().__init__()
            self.save_hyperparameters(logger=logger)

    model = TestModel("pytorch", "lightning")
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=0.1, limit_val_batches=0.1, num_sanity_val_steps=0
    )
    trainer.fit(model)

    if logger:
        log_hyperparams_mock.assert_called()
    else:
        log_hyperparams_mock.assert_not_called()


@patch("pytorch_lightning.loggers.tensorboard.TensorBoardLogger.log_hyperparams")
def test_log_hyperparams_key_collision(log_hyperparams_mock, tmpdir):
    class TestModel(BoringModel):
        def __init__(self, hparams: Dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters(hparams)

    class TestDataModule(BoringDataModule):
        def __init__(self, hparams: Dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters(hparams)

    class _Test:
        ...

    same_params = {1: 1, "2": 2, "three": 3.0, "test": _Test(), "4": torch.tensor(4)}
    model = TestModel(same_params)
    dm = TestDataModule(same_params)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # there should be no exceptions raised for the same key/value pair in the hparams of both
    # the lightning module and data module
    trainer.fit(model)

    obj_params = deepcopy(same_params)
    obj_params["test"] = _Test()
    model = TestModel(same_params)
    dm = TestDataModule(obj_params)
    trainer.fit(model)

    diff_params = deepcopy(same_params)
    diff_params.update({1: 0, "test": _Test()})
    model = TestModel(same_params)
    dm = TestDataModule(diff_params)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(MisconfigurationException, match="Error while merging hparams"):
        trainer.fit(model, dm)

    tensor_params = deepcopy(same_params)
    tensor_params.update({"4": torch.tensor(3)})
    model = TestModel(same_params)
    dm = TestDataModule(tensor_params)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(MisconfigurationException, match="Error while merging hparams"):
        trainer.fit(model, dm)
