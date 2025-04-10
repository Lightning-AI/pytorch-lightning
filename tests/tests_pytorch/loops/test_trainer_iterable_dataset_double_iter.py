import multiprocessing as mp
import os
from collections.abc import Iterator
from queue import Queue

import numpy as np
from torch.utils.data import DataLoader, IterableDataset

from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class QueueDataset(IterableDataset):
    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.queue = queue

    def __iter__(self) -> Iterator:
        for _ in range(5):
            tensor, _ = self.queue.get(timeout=5)
            yield tensor


def create_queue():
    q = mp.Queue()
    arr = np.random.random([1, 32]).astype(np.float32)
    for ind in range(10):
        q.put((arr, ind))
    return q


def train_model(queue, maxEpochs, ckptPath):
    dataloader = DataLoader(QueueDataset(queue), num_workers=1, batch_size=None, persistent_workers=True)
    trainer = Trainer(max_epochs=maxEpochs, enable_progress_bar=False, devices=1)
    if os.path.exists(ckptPath):
        trainer.fit(BoringModel(), dataloader, ckpt_path=ckptPath)
    else:
        trainer.fit(BoringModel(), dataloader)
        trainer.save_checkpoint(ckptPath)
    return trainer


def test_training():
    """Test that reproduces issue in calling iter twice on a queue-based IterableDataset leads to Queue Empty errors
    when resuming from a checkpoint."""
    queue = create_queue()

    ckpt_path = "model.ckpt"
    trainer = train_model(queue, 1, ckpt_path)
    assert trainer is not None

    assert os.path.exists(ckpt_path), "Checkpoint file wasn't created"

    ckpt_size = os.path.getsize(ckpt_path)
    assert ckpt_size > 0, f"Checkpoint file is empty (size: {ckpt_size} bytes)"

    trainer = train_model(queue, 2, ckpt_path)
    assert trainer is not None
