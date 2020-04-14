import os
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import tests.base.utils as tutils

from pytorch_lightning import Trainer, LightningModule
from tests.base.datasets import TestingMNIST


class ParityMNIST(LightningModule):

    def __init__(self):
        super(ParityMNIST, self).__init__()
        self.c_d1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.3)
        self.c_d2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(TestingMNIST(train=True,
                                       download=True,
                                       num_samples=500,
                                       digits=list(range(5))),
                          batch_size=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_pytorch_parity(tmpdir):
    """
    Verify that the same  pytorch and lightning models achieve the same results
    :param tmpdir:
    :return:
    """
    num_epochs = 2
    num_rums = 3
    lightning_outs, pl_times = lightning_loop(ParityMNIST, num_rums, num_epochs)
    manual_outs, pt_times = vanilla_loop(ParityMNIST, num_rums, num_epochs)

    # make sure the losses match exactly  to 5 decimal places
    for pl_out, pt_out in zip(lightning_outs, manual_outs):
        np.testing.assert_almost_equal(pl_out, pt_out, 5)

    tutils.assert_speed_parity(pl_times, pt_times, num_epochs)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def vanilla_loop(MODEL, num_runs=10, num_epochs=10):
    """
    Returns an array with the last loss from each epoch for each run
    """
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    errors = []
    times = []

    for i in range(num_runs):
        time_start = time.perf_counter()

        # set seed
        seed = i
        set_seed(seed)

        # init model parts
        model = MODEL()
        dl = model.train_dataloader()
        optimizer = model.configure_optimizers()

        # model to GPU
        model = model.to(device)

        epoch_losses = []
        for epoch in range(num_epochs):

            # run through full training set
            for j, batch in enumerate(dl):
                x, y = batch
                x = x.cuda(0)
                y = y.cuda(0)
                batch = (x, y)

                loss_dict = model.training_step(batch, j)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # track last epoch loss
            epoch_losses.append(loss.item())

        time_end = time.perf_counter()
        times.append(time_end - time_start)

        errors.append(epoch_losses[-1])

    return errors, times


def lightning_loop(MODEL, num_runs=10, num_epochs=10):
    errors = []
    times = []

    for i in range(num_runs):
        time_start = time.perf_counter()

        # set seed
        seed = i
        set_seed(seed)

        # init model parts
        model = MODEL()
        trainer = Trainer(
            max_epochs=num_epochs,
            show_progress_bar=False,
            weights_summary=None,
            gpus=1,
            early_stop_callback=False,
            checkpoint_callback=False
        )
        trainer.fit(model)

        final_loss = trainer.running_loss.last().item()
        errors.append(final_loss)

        time_end = time.perf_counter()
        times.append(time_end - time_start)

    return errors, times
