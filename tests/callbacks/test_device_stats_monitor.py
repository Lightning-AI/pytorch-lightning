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
from typing import Dict, Optional
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks.device_stats_monitor import _prefix_metric_keys
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import _CPU_PERCENT, _CPU_SWAP_PERCENT, _CPU_VM_PERCENT
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

CPU_METRIC_KEYS = [
    _CPU_VM_PERCENT,
    _CPU_SWAP_PERCENT,
    _CPU_PERCENT,
]


@RunIf(min_gpus=1)
def test_device_stats_gpu_from_torch(tmpdir):
    """Test GPU stats are logged using a logger."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = [
                "allocated_bytes.all.freed",
                "inactive_split.all.peak",
                "reserved_bytes.large_pool.peak",
            ]
            for f in fields:
                assert any(f in h for h in metrics)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@RunIf(max_torch="1.7")
@RunIf(min_gpus=1)
def test_device_stats_gpu_from_nvidia(tmpdir):
    """Test GPU stats are logged using a logger with Pytorch < 1.8.0."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor(cpu_stats=False)

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = [
                "utilization.gpu",
                "memory.used",
                "memory.free",
                "utilization.memory",
            ]
            for f in fields:
                assert any(f in h for h in metrics)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@pytest.mark.parametrize("cpu_stats", [None, True, False])
@RunIf(min_gpus=1, min_torch="1.8")
def test_device_stats_gpu_from_torch_toggle_cpu(tmpdir, cpu_stats):
    """Test only CPU stats can be enabled/disabled when using GPU."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor(cpu_stats=cpu_stats)

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            # Just need to check one of the GPU metrics to make sure
            # the GPU metrics are logged
            fields = [
                "active.all.allocated",
            ]

            # If cpu stats, also check CPU metric keys are logged
            fields = CPU_METRIC_KEYS if cpu_stats or cpu_stats is None else []
            for f in fields:
                assert any(f in h for h in metrics)

                # If not cpu stats, make sure CPU metric keys aren't logged
                if cpu_stats is not None and not cpu_stats:
                    assert not any(f in h for h in metrics), "CPU Stats should not be included"

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@pytest.mark.parametrize("cpu_stats", [None, True, False])
def test_device_stats_cpu(tmpdir, cpu_stats):
    """Test CPU stats are logged when no accelerator is used."""
    model = BoringModel()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = CPU_METRIC_KEYS if cpu_stats or cpu_stats is None else []
            for f in fields:
                assert any(f in h for h in metrics)

                # If not cpu stats, make sure CPU metric keys aren't logged
                if cpu_stats is not None and not cpu_stats:
                    assert not any(f in h for h in metrics), "CPU Stats should not be included"

    device_stats = DeviceStatsMonitor(cpu_stats=cpu_stats)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        log_every_n_steps=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@mock.patch("pytorch_lightning.accelerators.cpu.get_cpu_process_metrics")
@mock.patch("pytorch_lightning.callbacks.device_stats_monitor.get_cpu_process_metrics")
def test_device_stats_cpu_queried_once(cpu_metrics_device_stats_mock, cpu_metrics_cpu_accelerator_mock, tmpdir):
    """Make sure that get_cpu_process_metrics is only queried once if the accelerator is CPU and cpu_stats=True."""
    model = BoringModel()

    device_stats = DeviceStatsMonitor(cpu_stats=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        log_every_n_steps=1,
        callbacks=[device_stats],
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)

    assert cpu_metrics_device_stats_mock.call_count == 0  # called inside DeviceStatsMonitor
    assert cpu_metrics_cpu_accelerator_mock.call_count == 2  # called inside CPUAccelerator


@RunIf(tpu=True)
def test_device_stats_monitor_tpu(tmpdir):
    """Test TPU stats are logged using a logger."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]
            for f in fields:
                assert any(f in h for h in metrics)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        accelerator="tpu",
        devices=8,
        log_every_n_steps=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


def test_device_stats_monitor_no_logger(tmpdir):
    """Test DeviceStatsMonitor with no logger in Trainer."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[device_stats],
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    with pytest.raises(MisconfigurationException, match="`Trainer` that has no logger."):
        trainer.fit(model)


def test_prefix_metric_keys(tmpdir):
    """Test that metric key names are converted correctly."""
    metrics = {"1": 1.0, "2": 2.0, "3": 3.0}
    prefix = "foo"
    separator = "."
    converted_metrics = _prefix_metric_keys(metrics, prefix, separator)
    assert converted_metrics == {"foo.1": 1.0, "foo.2": 2.0, "foo.3": 3.0}
