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

"""Here is an example of `Lightning Fault Tolerant Automatic`.

Find the documentation: https://pytorch-lightning.readthedocs.io/en/stable/advanced/fault_tolerant_training.html

RUN WITHOUT FAILURE:

    1. Launch `python pl_examples/fault_tolerant/automatic.py`.
        - You should see `[-1.1343,  0.0186]` in the logs.

RUN WITH SIMULATED FAILURE:

    1. Launch `python pl_examples/fault_tolerant/automatic.py --emulate_kill_signal`.
        - You should see `kill -SIGTERM {PID}` in the logs.
    2. Run this command within another terminal.
        - You should see `Received signal 15. Saving a fault-tolerant checkpoint and terminating.` in the logs.
    3. Launch `python pl_examples/fault_tolerant/automatic.py --emulate_kill_signal` again.
        - You should see `Restored all states from the checkpoint file at ./.pl_auto_save.ckpt`
        - And you should see `[-1.1343,  0.0186]` in the logs.

    To restart the process, just run `rm .pl_auto_save.ckpt` to delete the auto restart checkpoint.

This example shows that the weights trained with failure matches the weight trained without failure,
thus the training has been properly resumed whilst being fully reproducible.

Used PyTorch 1.7.1.
"""

import os
import random as python_random
from argparse import ArgumentParser
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import _logger as log
from pytorch_lightning import LightningModule, seed_everything, Trainer


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


class SimpleMLP(LightningModule):
    def __init__(self, fail_on_step: int = -1):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.seen_batches = []
        self.fail_on_step = fail_on_step

    def training_step(self, batch, batch_idx):
        if self.global_step == self.fail_on_step:
            log.info(
                f"READY TO BE KILLED WITH SIGTERM SIGNAL. " f"Run `kill -SIGTERM {os.getpid()}` in another terminal."
            )
            # this line is used to wait for you to send the signal to exit gracefully.
            while not self.trainer._terminate_gracefully:
                sleep(0.1)
        batch = batch["data"] if isinstance(batch, dict) else batch
        self.seen_batches.append(torch.stack(batch) if isinstance(batch, list) else batch)
        loss = sum(self.layer(b).sum() for b in batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomGetItemDataset(3, 1))


def _run_training(default_root_dir=".", max_epochs=3, fail_on_step: int = -1, ckpt_path=None):
    model = SimpleMLP(fail_on_step=fail_on_step)
    trainer = Trainer(default_root_dir=default_root_dir, max_epochs=max_epochs)
    trainer.fit(model, ckpt_path=ckpt_path)
    return model.seen_batches, model.parameters()


def main(args):
    seed_everything(42)
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "automatic"  # active fault tolerant automatic

    ckpt_path = ".pl_auto_save.ckpt"
    auto_restart_ckpt_path_exists = os.path.exists(ckpt_path)
    if args.emulate_kill_signal:
        fail_on_step = -1 if auto_restart_ckpt_path_exists else 4
        completed_batches = 4 if auto_restart_ckpt_path_exists else 5
    else:
        fail_on_step = -1
        completed_batches = 9

    complete_batches, weights = _run_training(fail_on_step=fail_on_step)
    assert len(complete_batches) == completed_batches

    if not auto_restart_ckpt_path_exists and args.emulate_kill_signal:
        assert os.path.exists(ckpt_path)

    if auto_restart_ckpt_path_exists or not args.emulate_kill_signal:
        log.info([w for w in weights])


if __name__ == "__main__":
    parser = ArgumentParser(description="Fault Tolerant Under Signal Example")
    parser.add_argument(
        "--emulate_kill_signal",
        action="store_true",
        help="Whether you should gracefully kill the process with a `SIGTERM` signal.",
    )
    main(parser.parse_args())
