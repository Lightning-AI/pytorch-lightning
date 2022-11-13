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
import multiprocessing
import time
from typing import List, Tuple

import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, IterDataPipe

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.core.module import LightningModule


class FinallyCallCountingIterDataPipe(IterDataPipe):
    def __init__(self, assert_counts: List[Tuple[int, "FinallyCallCountingIterDataPipe"]] = []):
        self.assert_counts = assert_counts
        self._len = 128
        self.lock = multiprocessing.Lock()
        self.finally_called_n_times = multiprocessing.Manager().Value("i", 0)

    def __iter__(self):
        time.sleep(0.5)  # A deadline for the previous iterators to be cleaned up. Could be replaced with
        # some clever synchronization logic, but then the test would hang on failure.
        for expected_count, dp in self.assert_counts:
            with dp.lock:
                # >= is needed here to pass the check on the second epoch
                assert dp.finally_called_n_times.value >= expected_count
        try:
            yield from range(self._len)
        finally:
            with self.lock:
                self.finally_called_n_times.value += 1

    def __len__(self):
        return self._len


def test_evaluation_loop_disposes_of_iterator():
    val_data_pipe = FinallyCallCountingIterDataPipe()
    # When the train loop starts, we expect the iterator used by the sanity check to already be cleaned up
    train_data_pipe = FinallyCallCountingIterDataPipe([(4, val_data_pipe)])

    class TestModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = Linear(1, 1)

        def forward(self, *args, **kwargs):
            return self.layer(torch.zeros((1,)))

        def training_step(self, batch, batch_idx):
            return self.forward(batch)

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            return self.forward(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    class TestDataModule(LightningDataModule):
        def train_dataloader(self):
            return DataLoader(train_data_pipe, batch_size=4, num_workers=4)

        def val_dataloader(self):
            return DataLoader(val_data_pipe, batch_size=4, num_workers=4)

    model = TestModel()
    trainer = Trainer(max_epochs=2)
    data_module = TestDataModule()
    trainer.fit(model, datamodule=data_module)
    time.sleep(0.5)  # A deadline for the previous iterators to be cleaned up. Could be replaced with
    # some clever synchronization logic, but then the test would hang on failure.

    # 2 epochs, 4 processes => 8 disposed iterators
    assert train_data_pipe.finally_called_n_times.value == 8
    # 2 epochs, 4 processes + 1 sanity check => 12 disposed iterators
    assert val_data_pipe.finally_called_n_times.value == 12
