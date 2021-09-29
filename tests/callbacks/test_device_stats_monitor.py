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
from unittest.mock import patch

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_torch="1.8")
@RunIf(min_gpus=1)
def test_device_stats_gpu_from_torch(tmpdir):
    """Test GPU stats are logged using a logger with Pytorch >= 1.8.0."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    logger = CSVLogger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=1,
        gpus=1,
        callbacks=[device_stats],
        logger=logger,
        checkpoint_callback=False,
        enable_progress_bar=False,
    )

    # This line is needed to initialize cuda context, so that `torch.cuda.memory_stats` returns something.
    # See https://github.com/pytorch/pytorch/issues/65793
    _ = torch.randn(1, device="cuda")
    with patch.object(CSVLogger, "log_metrics") as mock_log_method:
        device_stats.on_train_batch_start(trainer, model, None, None, None)

    metrics_dict = mock_log_method.call_args.args[0]

    fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]

    for f in fields:
        assert any(f in h for h in metrics_dict.keys())


@RunIf(max_torch="1.7")
@RunIf(min_gpus=1)
def test_device_stats_gpu_from_nvidia(tmpdir):
    """Test GPU stats are logged using a logger with Pytorch < 1.8.0."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    logger = CSVLogger(tmpdir)
    log_every_n_steps = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=log_every_n_steps,
        gpus=1,
        callbacks=[device_stats],
        logger=logger,
        checkpoint_callback=False,
        enable_progress_bar=False,
    )

    with patch.object(CSVLogger, "log_metrics") as mock_log_method:
        device_stats.on_train_batch_start(trainer, model, None, None, None)

    metrics_dict = mock_log_method.call_args.args[0]

    fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]

    for f in fields:
        assert any(f in h for h in metrics_dict.keys())


@RunIf(tpu=True)
def test_device_stats_monitor_tpu(tmpdir):
    """Test TPU stats are logged using a logger."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    logger = CSVLogger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        tpu_cores=8,
        callbacks=[device_stats],
        logger=logger,
        checkpoint_callback=False,
        enable_progress_bar=False,
    )

    with patch.object(CSVLogger, "log_metrics") as mock_log_method:
        device_stats.on_train_batch_start(trainer, model, None, None, None)

    metrics_dict = mock_log_method.call_args.args[0]

    fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]

    for f in fields:
        assert any(f in h for h in metrics_dict.keys())


def test_device_stats_monitor_no_logger(tmpdir):
    """Test DeviceStatsMonitor with no logger in Trainer."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[device_stats],
        max_epochs=1,
        logger=False,
        checkpoint_callback=False,
        enable_progress_bar=False,
    )

    with pytest.raises(MisconfigurationException, match="Trainer that has no logger."):
        trainer.fit(model)
