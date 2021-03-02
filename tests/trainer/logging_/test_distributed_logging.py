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

from unittest import mock

from pytorch_lightning import Trainer
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class TestModel(BoringModel):

    def on_pretrain_routine_end(self) -> None:
        with mock.patch('pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics') as m:
            self.trainer.logger_connector.log_metrics({'a': 2}, {})
            logged_times = m.call_count
            expected = int(self.trainer.is_global_zero)
            msg = f'actual logger called from non-global zero, logged_times: {logged_times}, expected: {expected}'
            assert logged_times == expected, msg


@RunIf(skip_windows=True)
def test_global_zero_only_logging_ddp_cpu(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        accelerator='ddp_cpu',
        num_processes=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)


@RunIf(min_gpus=2)
def test_global_zero_only_logging_ddp_spawn(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        accelerator='ddp_spawn',
        gpus=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)
