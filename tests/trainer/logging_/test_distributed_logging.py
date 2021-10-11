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
from typing import Any, Dict, Optional, Union
from unittest.mock import Mock

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class AllRankLogger(LightningLoggerBase):
    """Logger to test all-rank logging (i.e. not just rank 0).

    Logs are saved to local variable `logs`.
    """

    def __init__(self):
        super().__init__()
        self.logs = {}
        self.exp = object()

    def experiment(self) -> Any:
        return self.exp

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.logs.update(metrics)

    def version(self) -> Union[int, str]:
        return 1

    def name(self) -> str:
        return "AllRank"

    def log_hyperparams(self, *args, **kwargs) -> None:
        pass


class TestModel(BoringModel):
    log_name = "rank-{rank}"

    def on_train_start(self):
        self.log(self.log_name.format(rank=self.local_rank), 0)

    def on_train_end(self):
        assert self.log_name.format(rank=self.local_rank) in self.logger.logs, "Expected rank to be logged"


@RunIf(skip_windows=True)
def test_all_rank_logging_ddp_cpu(tmpdir):
    """Check that all ranks can be logged from."""
    model = TestModel()
    all_rank_logger = AllRankLogger()
    trainer = Trainer(
        accelerator="ddp_cpu",
        num_processes=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        logger=all_rank_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model)


@RunIf(min_gpus=2)
def test_all_rank_logging_ddp_spawn(tmpdir):
    """Check that all ranks can be logged from."""
    model = TestModel()
    all_rank_logger = AllRankLogger()
    model.training_epoch_end = None
    trainer = Trainer(
        accelerator="ddp_spawn",
        gpus=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        logger=all_rank_logger,
        weights_summary=None,
    )
    trainer.fit(model)


def test_first_logger_call_in_subprocess(tmpdir):
    """Test that the Trainer does not call the logger too early.

    Only when the worker processes are initialized do we have access to the rank and know which one is the main process.
    """

    class LoggerCallsObserver(Callback):
        def on_fit_start(self, trainer, pl_module):
            # this hook is executed directly before Trainer.pre_dispatch
            # logger should not write any logs until this point
            assert not trainer.logger.method_calls
            assert not os.listdir(trainer.logger.save_dir)

        def on_train_start(self, trainer, pl_module):
            assert trainer.logger.method_call
            trainer.logger.log_hyperparams.assert_called_once()
            trainer.logger.log_graph.assert_called_once()

    logger = Mock()
    logger.version = "0"
    logger.name = "name"
    logger.save_dir = tmpdir

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        logger=logger,
        callbacks=[LoggerCallsObserver()],
    )
    trainer.fit(model)


def test_logger_after_fit_predict_test_calls(tmpdir):
    """Make sure logger outputs are finalized after fit, prediction, and test calls."""

    class BufferLogger(LightningLoggerBase):
        def __init__(self):
            super().__init__()
            self.buffer = {}
            self.logs = {}

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            self.buffer.update(metrics)

        def finalize(self, status: str) -> None:
            self.logs.update(self.buffer)
            self.buffer = {}

        @property
        def experiment(self) -> Any:
            return None

        @property
        def version(self) -> Union[int, str]:
            return 1

        @property
        def name(self) -> str:
            return "BufferLogger"

        def log_hyperparams(self, *args, **kwargs) -> None:
            return None

    class LoggerCallsObserver(Callback):
        def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            trainer.logger.log_metrics({"fit": 1})

        def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            trainer.logger.log_metrics({"validate": 1})

        def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            trainer.logger.log_metrics({"predict": 1})

        def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            trainer.logger.log_metrics({"test": 1})

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        logger=BufferLogger(),
        callbacks=[LoggerCallsObserver()],
    )

    assert not trainer.logger.logs
    trainer.fit(model)
    assert trainer.logger.logs == {"fit": 1, "validate": 1}
    trainer.test(model)
    assert trainer.logger.logs == {"fit": 1, "validate": 1, "test": 1}
    trainer.predict(model)
    assert trainer.logger.logs == {"fit": 1, "validate": 1, "test": 1, "predict": 1}
