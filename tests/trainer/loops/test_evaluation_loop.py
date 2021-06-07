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

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


@mock.patch("pytorch_lightning.trainer.evaluation_loop.EvaluationLoop.on_evaluation_epoch_end")
def test_on_evaluation_epoch_end(eval_epoch_end_mock, tmpdir):
    """
    Tests that `on_evaluation_epoch_end` is called
    for `on_validation_epoch_end` and `on_test_epoch_end` hooks
    """
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        weights_summary=None,
    )

    trainer.fit(model)
    # sanity + 2 epochs
    assert eval_epoch_end_mock.call_count == 3

    trainer.test()
    # sanity + 2 epochs + called once for test
    assert eval_epoch_end_mock.call_count == 4


@mock.patch(
    "pytorch_lightning.trainer.connectors.logger_connector.logger_connector.LoggerConnector.get_evaluate_epoch_results"
)
def test_log_epoch_metrics_before_on_evaluation_end(get_evaluate_epoch_results_mock, tmpdir):
    """Test that the epoch metrics are logged before the `on_evalutaion_end` hook is fired"""
    order = []
    get_evaluate_epoch_results_mock.side_effect = lambda: order.append("log_epoch_metrics")

    class LessBoringModel(BoringModel):

        def on_validation_end(self):
            order.append("on_validation_end")
            super().on_validation_end()

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        weights_summary=None,
        num_sanity_val_steps=0,
    )
    trainer.fit(LessBoringModel())

    assert order == ["log_epoch_metrics", "on_validation_end"]


@RunIf(min_gpus=1)
def test_memory_consumption_validation(tmpdir):
    """Test that the training batch is no longer in GPU memory when running validation"""

    class BoringLargeBatchModel(BoringModel):

        @property
        def num_params(self):
            return sum(p.numel() for p in self.parameters())

        def train_dataloader(self):
            # batch target memory >= 100x boring_model size
            batch_size = self.num_params * 100 // 32 + 1
            return DataLoader(RandomDataset(32, 5000), batch_size=batch_size)

        def val_dataloader(self):
            return self.train_dataloader()

        def training_step(self, batch, batch_idx):
            # there is a batch and the boring model, but not two batches on gpu, assume 32 bit = 4 bytes
            lower = 101 * self.num_params * 4
            upper = 201 * self.num_params * 4
            assert lower < torch.cuda.memory_allocated(0) < upper
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            # there is a batch and the boring model, but not two batches on gpu, assume 32 bit = 4 bytes
            lower = 101 * self.num_params * 4
            upper = 201 * self.num_params * 4
            assert lower < torch.cuda.memory_allocated(0) < upper
            return super().validation_step(batch, batch_idx)

    torch.cuda.empty_cache()
    trainer = Trainer(
        gpus=1,
        default_root_dir=tmpdir,
        fast_dev_run=2,
        move_metrics_to_cpu=True,
        weights_summary=None,
    )
    trainer.fit(BoringLargeBatchModel())
