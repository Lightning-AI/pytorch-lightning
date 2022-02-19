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
from typing import Optional

import pytest

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class CustomDeprecatedLogger(LightningLoggerBase):
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


def test_lightning_logger_base_deprecation_warning():
    with pytest.deprecated_call(match="The `pl.loggers.base.LightningLoggerBase` is deprecated."):
        CustomDeprecatedLogger()
