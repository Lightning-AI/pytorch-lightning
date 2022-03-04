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

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from tests.loggers.test_base import CustomLogger


def test_trainer_loggers_property():
    """Test for correct initialization of loggers in Trainer."""
    logger1 = CustomLogger()
    logger2 = CustomLogger()

    # trainer.loggers should be a copy of the input list
    trainer = Trainer(logger=[logger1, logger2])

    assert trainer.loggers == [logger1, logger2]

    # trainer.loggers should create a list of size 1
    trainer = Trainer(logger=logger1)

    assert trainer.logger == logger1
    assert trainer.loggers == [logger1]

    # trainer.loggers should be an empty list
    trainer = Trainer(logger=False)

    assert trainer.logger is None
    assert trainer.loggers == []

    # trainer.loggers should be a list of size 1 holding the default logger
    trainer = Trainer(logger=True)

    assert trainer.loggers == [trainer.logger]
    assert isinstance(trainer.logger, TensorBoardLogger)


def test_trainer_loggers_setters():
    """Test the behavior of setters for trainer.logger and trainer.loggers."""
    logger1 = CustomLogger()
    logger2 = CustomLogger()
    with pytest.deprecated_call(match="`LoggerCollection` is deprecated in v1.6"):
        logger_collection = LoggerCollection([logger1, logger2])
    with pytest.deprecated_call(match="`LoggerCollection` is deprecated in v1.6"):
        logger_collection_2 = LoggerCollection([logger2])

    trainer = Trainer()
    assert type(trainer.logger) == TensorBoardLogger
    assert trainer.loggers == [trainer.logger]

    # Test setters for trainer.logger
    trainer.logger = logger1
    assert trainer.logger == logger1
    assert trainer.loggers == [logger1]

    trainer.logger = logger_collection
    assert trainer.logger._logger_iterable == logger_collection._logger_iterable
    assert trainer.loggers == [logger1, logger2]

    # LoggerCollection of size 1 should result in trainer.logger becoming the contained logger.
    trainer.logger = logger_collection_2
    assert trainer.logger == logger2
    assert trainer.loggers == [logger2]

    trainer.logger = None
    assert trainer.logger is None
    assert trainer.loggers == []

    # Test setters for trainer.loggers
    trainer.loggers = [logger1, logger2]
    assert trainer.loggers == [logger1, logger2]
    assert trainer.logger._logger_iterable == logger_collection._logger_iterable

    trainer.loggers = [logger1]
    assert trainer.loggers == [logger1]
    assert trainer.logger == logger1

    trainer.loggers = []
    assert trainer.loggers == []
    assert trainer.logger is None

    trainer.loggers = None
    assert trainer.loggers == []
    assert trainer.logger is None
