import os
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor(tmpdir):
    """
    Test GPU stats are logged using a logger.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor()
    logger = CSVLogger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        callbacks=[gpu_stats],
        logger=logger
    )

    results = trainer.fit(model)
    assert results

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv, 'r') as fp:
        lines = fp.readlines()

    header = lines[0].split()

    fields = [
        'GPU_utilization.gpu',
        'GPU_memory.used',
        'GPU_memory.free',
        'GPU_utilization.memory'
    ]

    for f in fields:
        assert any([f in h for h in header])


@pytest.mark.skipif(torch.cuda.is_available(), reason="test requires CPU machine")
def test_gpu_stats_monitor_cpu_machine(tmpdir):
    """
    Test GPUStatsMonitor on CPU machine.
    """
    with pytest.raises(MisconfigurationException, match='nvidia driver is not installed'):
        gpu_stats = GPUStatsMonitor()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor_no_logger(tmpdir):
    """
    Test GPUStatsMonitor with no logger in Trainer.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_epochs=1,
        gpus=1,
        logger=None
    )

    with pytest.raises(MisconfigurationException, match='Trainer that has no logger.'):
        trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor_no_gpu_warning(tmpdir):
    """
    Test GPUStatsMonitor raises a warning when not training on GPU device.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_steps=1,
        gpus=None
    )

    with pytest.warns(RuntimeWarning, match='not running on GPU. Logged utilization will be independent'):
        trainer.fit(model)
