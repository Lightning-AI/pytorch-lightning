# Copyright The Lightning AI team.
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
#
# This test tests the resuming of training from a checkpoint file using an IterableDataset.
# And contains code mentioned in the issue: #19427.
# Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/19427
import multiprocessing as mp
import os
from collections.abc import Iterator
from pathlib import Path
from queue import Queue

import numpy as np
from torch.utils.data import DataLoader, IterableDataset

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class QueueDataset(IterableDataset):
    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.queue = queue

    def __iter__(self) -> Iterator:
        for _ in range(5):
            tensor, _ = self.queue.get(timeout=5)
            yield tensor


def create_queue() -> Queue:
    q = mp.Queue()
    arr = np.random.random([1, 32]).astype(np.float32)
    for ind in range(20):
        q.put((arr, ind))
    return q


def train_model(queue: Queue, max_epochs: int, ckpt_path: Path) -> Trainer:
    dataloader = DataLoader(QueueDataset(queue), num_workers=1, batch_size=None, persistent_workers=True)
    trainer = Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        devices=1,
        logger=False,
    )
    if ckpt_path.exists():
        trainer.fit(BoringModel(), dataloader, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(BoringModel(), dataloader)
        trainer.save_checkpoint(str(ckpt_path))
    return trainer


def test_resume_training_with(tmp_path):
    """Test resuming training from checkpoint file using a IterableDataset."""
    queue = create_queue()
    max_epoch = 2
    ckpt_path = tmp_path / "model.ckpt"
    trainer = train_model(queue, max_epoch, ckpt_path)
    assert trainer is not None

    assert os.path.exists(ckpt_path), f"Checkpoint file '{ckpt_path}' wasn't created"

    ckpt_size = os.path.getsize(ckpt_path)
    assert ckpt_size > 0, f"Checkpoint file is empty (size: {ckpt_size} bytes)"

    trainer = train_model(queue, max_epoch + 2, ckpt_path)
    assert trainer is not None
