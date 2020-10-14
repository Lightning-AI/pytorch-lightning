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
import torch
from tests.base import BoringModel
import platform
from distutils.version import LooseVersion
from pytorch_lightning import Trainer, Callback
from unittest import mock


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
