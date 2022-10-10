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
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.loggers import CSVLogger


def test_v2_0_0_unsupported_agg_and_log_metrics(tmpdir):
    class AggAndLogMetricsLogger(CSVLogger):
        def agg_and_log_metrics(self, metrics, step):
            pass

    model = BoringModel()
    logger = AggAndLogMetricsLogger(tmpdir)

    trainer = Trainer(logger=logger)

    with pytest.raises(
        RuntimeError,
        match="`AggAndLogMetricsLogger.agg_and_log_metrics` was deprecated in v1.6 and is no longer supported",
    ):
        trainer.fit(model)


def test_v2_0_0_unsupported_update_agg_funcs(tmpdir):
    class UpdateAggFuncsLogger(CSVLogger):
        def update_agg_funcs(self, metrics, step):
            pass

    model = BoringModel()
    logger = UpdateAggFuncsLogger(tmpdir)

    trainer = Trainer(logger=logger)

    with pytest.raises(
        RuntimeError,
        match="`AggregationOverrideLogger.update_agg_funcs` was deprecated in v1.6 and is no longer supported",
    ):
        trainer.fit(model)


def test_v2_0_0_unsupported_logger_collection_class():
    from pytorch_lightning.loggers.base import LoggerCollection

    with pytest.raises(RuntimeError, match="`LoggerCollection` was deprecated in v1.6 and removed in v1.8."):
        LoggerCollection(None)

    from pytorch_lightning.loggers.logger import LoggerCollection

    with pytest.raises(RuntimeError, match="`LoggerCollection` was deprecated in v1.6 and removed in v1.8."):
        LoggerCollection(None)
