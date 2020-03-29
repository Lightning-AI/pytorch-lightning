import numpy as np
import time

import torch

from pytorch_lightning import Trainer
from tests.base import (
    ParityMNIST,
)


def test_pytorch_parity(tmpdir):
    """
    Verify that the same pytorch and lightning models achieve the same results
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
            early_stop_callback=False
        )
        trainer.fit(model)

        final_loss = trainer.running_loss[-1]
        errors.append(final_loss)

        time_end = time.perf_counter()
        times.append(time_end - time_start)

    return errors, times
