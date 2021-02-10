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
from typing import Optional
from unittest.mock import MagicMock

import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, TensorBoardLogger
from pytorch_lightning.loggers.base import DummyExperiment, DummyLogger
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import rank_zero_only
from tests.helpers import BoringModel


def test_logger_collection():
    mock1 = MagicMock()
    mock2 = MagicMock()

    logger = LoggerCollection([mock1, mock2])

    assert logger[0] == mock1
    assert logger[1] == mock2

    assert logger.experiment[0] == mock1.experiment
    assert logger.experiment[1] == mock2.experiment

    assert logger.save_dir is None

    logger.update_agg_funcs({'test': np.mean}, np.sum)
    mock1.update_agg_funcs.assert_called_once_with({'test': np.mean}, np.sum)
    mock2.update_agg_funcs.assert_called_once_with({'test': np.mean}, np.sum)

    logger.agg_and_log_metrics({'test': 2.0}, 4)
    mock1.agg_and_log_metrics.assert_called_once_with({'test': 2.0}, 4)
    mock2.agg_and_log_metrics.assert_called_once_with({'test': 2.0}, 4)

    logger.close()
    mock1.close.assert_called_once()
    mock2.close.assert_called_once()


class CustomLogger(LightningLoggerBase):

    def __init__(self):
        super().__init__()
        self.hparams_logged = None
        self.metrics_logged = {}
        self.finalized = False

    @property
    def experiment(self):
        return 'test'

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
        """
        Return the root directory where experiment logs get saved, or `None` if the logger does not
        save data locally.
        """
        return None

    @property
    def name(self):
        return "name"

    @property
    def version(self):
        return "1"


def test_custom_logger(tmpdir):

    class CustomModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('train_loss', loss)
            return {"loss": loss}

    logger = CustomLogger()
    model = CustomModel()
    trainer = Trainer(
        max_steps=2,
        log_every_n_steps=1,
        logger=logger,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert logger.hparams_logged == model.hparams
    assert logger.metrics_logged != {}
    assert logger.finalized_status == "success"


def test_multiple_loggers(tmpdir):

    class CustomModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('train_loss', loss)
            return {"loss": loss}

    model = CustomModel()
    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(
        max_steps=2,
        log_every_n_steps=1,
        logger=[logger1, logger2],
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

    assert logger1.hparams_logged == model.hparams
    assert logger1.metrics_logged != {}
    assert logger1.finalized_status == "success"

    assert logger2.hparams_logged == model.hparams
    assert logger2.metrics_logged != {}
    assert logger2.finalized_status == "success"


def test_multiple_loggers_pickle(tmpdir):
    """Verify that pickling trainer with multiple loggers works."""

    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(logger=[logger1, logger2], )
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0}, 0)

    assert trainer2.logger[0].metrics_logged == {"acc": 1.0}
    assert trainer2.logger[1].metrics_logged == {"acc": 1.0}


def test_adding_step_key(tmpdir):

    class CustomTensorBoardLogger(TensorBoardLogger):

        def __init__(self, *args, **kwargs):
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


def test_with_accumulate_grad_batches():
    """Checks if the logging is performed once for `accumulate_grad_batches` steps."""

    class StoreHistoryLogger(CustomLogger):

        def __init__(self):
            super().__init__()
            self.history = {}

        @rank_zero_only
        def log_metrics(self, metrics, step):
            if step not in self.history:
                self.history[step] = {}
            self.history[step].update(metrics)

    logger = StoreHistoryLogger()

    np.random.seed(42)
    for i, loss in enumerate(np.random.random(10)):
        logger.agg_and_log_metrics({'loss': loss}, step=int(i / 5))

    assert logger.history == {0: {'loss': 0.5623850983416314}}
    logger.close()
    assert logger.history == {0: {'loss': 0.5623850983416314}, 1: {'loss': 0.4778883735637184}}


def test_dummyexperiment_support_indexing():
    experiment = DummyExperiment()
    assert experiment[0] == experiment


def test_dummylogger_support_indexing():
    logger = DummyLogger()
    assert logger[0] == logger


def test_np_sanitization():

    class CustomParamsLogger(CustomLogger):

        def __init__(self):
            super().__init__()
            self.logged_params = None

        @rank_zero_only
        def log_hyperparams(self, params):
            params = self._convert_params(params)
            params = self._sanitize_params(params)
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
