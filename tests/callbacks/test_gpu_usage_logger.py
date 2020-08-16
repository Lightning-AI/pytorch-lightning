import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GpuUsageLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
import tests.base.develop_utils as tutils


@pytest.mark.skipif(torch.cuda.is_available(), reason="test requires CPU machine")
def test_gpu_usage_logger_cpu_machine(tmpdir):
    """ Test GpuUsageLogger on CPU machine. """
    mode = EvalModelTemplate()

    with pytest.raises(MisconfigurationException, match='nvidia driver is not installed'):
        gpu_usage = GpuUsageLogger()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_usage_logger_no_logger(tmpdir):
    """ Test GpuUsageLogger with no logger in Trainer. """
    model = EvalModelTemplate()
    gpu_usage = GpuUsageLogger()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_usage],
        max_epochs=1,
        logger=None
    )

    with pytest.raises(MisconfigurationException, match='Trainer that has no logger.'):
        trainer.fit(model)
