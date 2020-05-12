import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tests.base.utils as tutils

from pytorch_lightning import Trainer, LightningModule, seed_everything


class AverageDataset(Dataset):
    def __init__(self, dataset_len=300, sequence_len=100):
        self.dataset_len = dataset_len
        self.sequence_len = sequence_len
        self.input_seq = torch.randn(dataset_len, sequence_len, 10)
        top, bottom = self.input_seq.chunk(2, -1)
        self.output_seq = top + bottom.roll(shifts=1, dims=-1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item]


class ParityRNN(LightningModule):
    def __init__(self):
        super(ParityRNN, self).__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.linear_out = nn.Linear(in_features=20, out_features=5)

    def forward(self, x):
        seq, last = self.rnn(x)
        return self.linear_out(seq)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(AverageDataset(), batch_size=30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_pytorch_parity(tmpdir):
    """
    Verify that the same  pytorch and lightning models achieve the same results
    :param tmpdir:
    :return:
    """
    num_epochs = 2
    num_rums = 3

    lightning_outs, pl_times = lightning_loop(ParityRNN, num_rums, num_epochs)
    manual_outs, pt_times = vanilla_loop(ParityRNN, num_rums, num_epochs)
    # make sure the losses match exactly  to 5 decimal places
    for pl_out, pt_out in zip(lightning_outs, manual_outs):
        np.testing.assert_almost_equal(pl_out, pt_out, 8)

    tutils.assert_speed_parity(pl_times, pt_times, num_epochs)


def vanilla_loop(MODEL, num_runs=10, num_epochs=10):
    """
    Returns an array with the last loss from each epoch for each run
    """
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    errors = []
    times = []

    torch.backends.cudnn.deterministic = True
    for i in range(num_runs):
        time_start = time.perf_counter()

        # set seed
        seed = i
        seed_everything(seed)

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
        seed_everything(seed)
        model = MODEL()

        # init model parts
        trainer = Trainer(
            max_epochs=num_epochs,
            progress_bar_refresh_rate=0,
            weights_summary=None,
            gpus=1,
            early_stop_callback=False,
            checkpoint_callback=False,
            distributed_backend='dp',
            deterministic=True,
        )
        trainer.fit(model)

        final_loss = trainer.running_loss.last().item()
        errors.append(final_loss)

        time_end = time.perf_counter()
        times.append(time_end - time_start)

    return errors, times
