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

"""Here are the steps to use this file to ensure your cluster is properly setup for Fault Tolerant Training.

1. Launch this script with `python tests/utilities/fault_tolerant.py`
2. Once ``READY TO BE KILLED WITH SIGTERM SIGNAL`` is detected within
   the Pod Logs, send a SIGTERM to this process. The process is waiting for it.
3. You should detect `.pl_auto_save.ckpt exists` within the Pod Logs.
4. Resume your instance with the same volumes and relaunch the same script.
5. The script should contain `[-1.0939, -0.4306]` within its Pod Logs.

The mode weights with the failure matches the weight without.
The training has been properly resumed and is fully reproduced.

Command: `python -m tests.utilities.fault_tolerant`
"""

# Note, this file is used to ensure Fault Tolerant is working as expected
import os
import random as python_random
from contextlib import suppress
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, seed_everything, Trainer


class Model(LightningModule):
    def __init__(self, fail_on_step: int = -1):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.seen_batches = []
        self.fail_on_step = fail_on_step

    def training_step(self, batch, batch_idx):
        if self.global_step == self.fail_on_step:
            print("READY TO BE KILLED WITH SIGTERM SIGNAL.")
            while not self.trainer._terminate_gracefully:
                sleep(0.00001)
            raise CustomException()
        batch = batch["data"] if isinstance(batch, dict) else batch
        self.seen_batches.append(torch.stack(batch) if isinstance(batch, list) else batch)
        loss = sum(self.layer(b).sum() for b in batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class CustomException(Exception):
    pass


class RandomGetItemDataset(Dataset):
    """A dataset with random elements generated using global rng from torch, numpy and python."""

    def __init__(self, length, size):
        self.size = size
        self.len = length

    def __getitem__(self, index):
        t = torch.rand(self.size)
        n = torch.from_numpy(np.random.rand(self.size))
        p = torch.tensor([python_random.random() for _ in range(self.size)])
        sample = (index + (t + n + p) / 10).float()
        return sample

    def __len__(self):
        return self.len


def _run_training(trainer_kwargs, dataset_classes, fail_on_step: int = -1, ckpt_path=None):
    seed_everything(1)
    train_dataloader = [
        DataLoader(dataset_class(3, 1), batch_size=1, num_workers=0) for dataset_class in dataset_classes
    ]
    train_dataloader = train_dataloader[0] if len(train_dataloader) == 1 else train_dataloader
    model = Model(fail_on_step=fail_on_step)
    trainer = Trainer(**trainer_kwargs)
    with suppress(CustomException):
        trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=ckpt_path)
    return model.seen_batches, model.parameters()


tmpdir = "/tmp/pl_fault_tolerant"

os.makedirs(tmpdir, exist_ok=True)

env_backup = os.environ.copy()

auto_restart_checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
auto_restart_checkpoint_path_exists = os.path.exists(auto_restart_checkpoint_path)

seed_everything(42)

os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"

dataset_classes = [RandomGetItemDataset]

trainer_kwargs = dict(
    default_root_dir=tmpdir,
    max_epochs=3,
    enable_progress_bar=False,
    enable_model_summary=False,
)

if auto_restart_checkpoint_path_exists:
    fail_on_step = -1
    completed_batches = 5
else:
    fail_on_step = 4
    completed_batches = 4

# Perform a failure
complete_batches, weights = _run_training(trainer_kwargs, dataset_classes, fail_on_step=fail_on_step)
assert len(complete_batches) == completed_batches

if not auto_restart_checkpoint_path_exists:
    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)
    print(".pl_auto_save.ckpt exists.")

if auto_restart_checkpoint_path_exists:
    print([w for w in weights])

os.environ.clear()
os.environ.update(env_backup)
