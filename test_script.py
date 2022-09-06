import os
import random as python_random
from argparse import ArgumentParser
from time import sleep

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger


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
            print(f"READY TO BE KILLED WITH SIGTERM SIGNAL. " f"Run `kill -SIGTERM {os.getpid()}` in another terminal.")
            # this line is used to wait for you to send the signal to exit gracefully.
            while not self.trainer._terminate_gracefully:
                sleep(0.1)
        batch = batch["data"] if isinstance(batch, dict) else batch
        self.seen_batches.append(torch.stack(batch) if isinstance(batch, list) else batch)
        loss = sum(self.layer(b).sum() for b in batch)
        self.log("loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomGetItemDataset(3, 1))


def _run_training(default_root_dir=".", max_epochs=3, fail_on_step: int = -1, ckpt_path=None, logger=True):
    model = SimpleMLP(fail_on_step=fail_on_step)
    trainer = Trainer(default_root_dir=default_root_dir, max_epochs=max_epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(model, ckpt_path=ckpt_path)
    wandb.finish()
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

    if args.use_tb:
        logger = True
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            id=args.wandb_run,
        )

    complete_batches, weights = _run_training(fail_on_step=fail_on_step, logger=logger)
    assert len(complete_batches) == completed_batches

    if not auto_restart_ckpt_path_exists and args.emulate_kill_signal:
        assert os.path.exists(ckpt_path)

    if auto_restart_ckpt_path_exists or not args.emulate_kill_signal:
        print([w for w in weights])


if __name__ == "__main__":
    parser = ArgumentParser(description="Fault Tolerant Under Signal Example")
    parser.add_argument(
        "--emulate_kill_signal",
        action="store_true",
        help="Whether you should gracefully kill the process with a `SIGTERM` signal.",
    )
    parser.add_argument(
        "--use_tb",
        action="store_true",
        help="Use TensorBoard instead of WandB.",
    )
    parser.add_argument(
        "-e",
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity.",
    )
    parser.add_argument(
        "-p",
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project.",
    )
    parser.add_argument(
        "-r",
        "--wandb_run",
        type=str,
        default=None,
        help="Wandb run.",
    )
    main(parser.parse_args())
