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

import numpy as np
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_gpus=1)
def test_gpu_stats_monitor(tmpdir):
    """
    Test GPU stats are logged using a logger.
    """
    model = BoringModel()
    gpu_stats = GPUStatsMonitor(intra_step_time=True)
    logger = CSVLogger(tmpdir)
    log_every_n_steps = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=log_every_n_steps,
        gpus=1,
        callbacks=[gpu_stats],
        logger=logger
    )

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    met_data = np.genfromtxt(path_csv, delimiter=',', names=True, deletechars='', replace_space=' ')

    batch_time_data = met_data['batch_time/intra_step (ms)']
    batch_time_data = batch_time_data[~np.isnan(batch_time_data)]
    assert batch_time_data.shape[0] == trainer.global_step // log_every_n_steps

    fields = [
        'utilization.gpu',
        'memory.used',
        'memory.free',
        'utilization.memory',
    ]

    for f in fields:
        assert any([f in h for h in met_data.dtype.names])


@pytest.mark.skipif(torch.cuda.is_available(), reason="test requires CPU machine")
def test_gpu_stats_monitor_cpu_machine(tmpdir):
    """
    Test GPUStatsMonitor on CPU machine.
    """
    with pytest.raises(MisconfigurationException, match='NVIDIA driver is not installed'):
        GPUStatsMonitor()


@RunIf(min_gpus=1)
def test_gpu_stats_monitor_no_logger(tmpdir):
    """
    Test GPUStatsMonitor with no logger in Trainer.
    """
    model = BoringModel()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_epochs=1,
        gpus=1,
        logger=False,
    )

    with pytest.raises(MisconfigurationException, match='Trainer that has no logger.'):
        trainer.fit(model)


@RunIf(min_gpus=1)
def test_gpu_stats_monitor_no_gpu_warning(tmpdir):
    """
    Test GPUStatsMonitor raises a warning when not training on GPU device.
    """
    model = BoringModel()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_steps=1,
        gpus=None,
    )

    with pytest.raises(MisconfigurationException, match='not running on GPU'):
        trainer.fit(model)


def test_gpu_stats_monitor_parse_gpu_stats():
    logs = GPUStatsMonitor._parse_gpu_stats('1,2', [[3, 4, 5], [6, 7]], [('gpu', 'a'), ('memory', 'b')])
    expected = {'gpu_id: 1/gpu (a)': 3, 'gpu_id: 1/memory (b)': 4, 'gpu_id: 2/gpu (a)': 6, 'gpu_id: 2/memory (b)': 7}
    assert logs == expected
