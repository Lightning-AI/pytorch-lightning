import time

import numpy as np
import pytest
import torch

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, seed_everything
from tests.base.models import ParityModuleMNIST, ParityModuleRNN


# ParityModuleMNIST runs with num_workers=1
@pytest.mark.parametrize('cls_model,max_diff', [
    (ParityModuleRNN, 0.05),
    (ParityModuleMNIST, 0.22)
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_pytorch_parity(tmpdir, cls_model, max_diff):
    """
    Verify that the same  pytorch and lightning models achieve the same results
    """
    num_epochs = 4
    num_rums = 3
    lightning_outs, pl_times = lightning_loop(cls_model, num_rums, num_epochs)
    manual_outs, pt_times = vanilla_loop(cls_model, num_rums, num_epochs)

    # make sure the losses match exactly  to 5 decimal places
    for pl_out, pt_out in zip(lightning_outs, manual_outs):
        np.testing.assert_almost_equal(pl_out, pt_out, 5)

    # the fist run initialize dataset (download & filter)
    tutils.assert_speed_parity_absolute(pl_times[1:], pt_times[1:],
                                        nb_epochs=num_epochs, max_diff=max_diff)


def vanilla_loop(cls_model, num_runs=10, num_epochs=10):
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
        model = cls_model()
        dl = model.train_dataloader()
        optimizer = model.configure_optimizers()

        # model to GPU
        model = model.to(device)

        epoch_losses = []
        # as the first run is skipped, no need to run it long
        for epoch in range(num_epochs if i > 0 else 1):

            # run through full training set
            for j, batch in enumerate(dl):
                batch = [x.to(device) for x in batch]
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


def lightning_loop(cls_model, num_runs=10, num_epochs=10):
    errors = []
    times = []

    for i in range(num_runs):
        time_start = time.perf_counter()

        # set seed
        seed = i
        seed_everything(seed)

        model = cls_model()
        # init model parts
        trainer = Trainer(
            # as the first run is skipped, no need to run it long
            max_epochs=num_epochs if i > 0 else 1,
            progress_bar_refresh_rate=0,
            weights_summary=None,
            gpus=1,
            checkpoint_callback=False,
            deterministic=True,
            logger=False,
            replace_sampler_ddp=False,
        )
        trainer.fit(model)

        final_loss = trainer.train_loop.running_loss.last().item()
        errors.append(final_loss)

        time_end = time.perf_counter()
        times.append(time_end - time_start)

    return errors, times
