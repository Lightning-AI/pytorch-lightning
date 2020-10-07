import pytest
import torch
import os
from tests.base import BoringModel
import platform
from distutils.version import LooseVersion
from pytorch_lightning import Trainer, Callback
from unittest import mock


os.environ['PL_DEV_DEBUG'] = '1'

class TestModel(BoringModel):

    def on_pretrain_routine_end(self) -> None:
        with mock.patch('pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics') as m:
            self.trainer.logger_connector.log_metrics({'a': 2}, {})
            logged_times = m.call_count
            expected = 1 if self.global_rank == 0 else 0
            assert logged_times == expected, 'actual logger called from non-global zero'


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif((platform.system() == "Darwin" and
                     LooseVersion(torch.__version__) < LooseVersion("1.3.0")),
                    reason="Distributed training is not supported on MacOS before Torch 1.3.0")
def test_global_zero_only_logging_ddp_cpu(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        distributed_backend='ddp_cpu',
        num_processes=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_global_zero_only_logging_ddp_spawn(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        distributed_backend='ddp_spawn',
        gpus=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)
