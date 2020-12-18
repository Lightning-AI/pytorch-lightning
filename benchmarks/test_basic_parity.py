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

import time

import numpy as np
import pytest
import torch
from tqdm import tqdm

from pytorch_lightning import seed_everything, Trainer
import tests.base.develop_utils as tutils
from tests.base.models import ParityModuleMNIST, ParityModuleRNN


# ParityModuleMNIST runs with num_workers=1
@pytest.mark.parametrize('cls_model,max_diff', [
    (ParityModuleRNN, 0.05),
    (ParityModuleMNIST, 0.25),  # todo: lower this thr
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_pytorch_parity(tmpdir, cls_model, max_diff: float, num_epochs: int = 4, num_runs: int = 3):
    """
    Verify that the same  pytorch and lightning models achieve the same results
    """
    lightning = lightning_loop(cls_model, num_runs, num_epochs)
    vanilla = vanilla_loop(cls_model, num_runs, num_epochs)

    # make sure the losses match exactly  to 5 decimal places
    for pl_out, pt_out in zip(lightning['losses'], vanilla['losses']):
        np.testing.assert_almost_equal(pl_out, pt_out, 5)

    # the fist run initialize dataset (download & filter)
    tutils.assert_speed_parity_absolute(
        lightning['durations'][1:], vanilla['durations'][1:], nb_epochs=num_epochs, max_diff=max_diff
    )


def vanilla_loop(cls_model, num_runs=10, num_epochs=10):
    """
    Returns an array with the last loss from each epoch for each run
    """
    hist_losses = []
    hist_durations = []

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    for i in tqdm(range(num_runs), desc=f'Vanilla PT with {cls_model.__name__}'):
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
        hist_durations.append(time_end - time_start)

        hist_losses.append(epoch_losses[-1])

    return {
        'losses': hist_losses,
        'durations': hist_durations,
    }


def lightning_loop(cls_model, num_runs=10, num_epochs=10):
    hist_losses = []
    hist_durations = []

    for i in tqdm(range(num_runs), desc=f'PT Lightning with {cls_model.__name__}'):
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
        hist_losses.append(final_loss)

        time_end = time.perf_counter()
        hist_durations.append(time_end - time_start)

    return {
        'losses': hist_losses,
        'durations': hist_durations,
    }
