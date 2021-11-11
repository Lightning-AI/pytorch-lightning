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
import pytest
import torch
from torch.utils.data.sampler import Sampler, SequentialSampler

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel, RandomDataset


def test_overfit_multiple_val_loaders(tmpdir):
    """Tests that overfit batches works with multiple val dataloaders."""
    val_dl_count = 2
    overfit_batches = 3

    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx):
            output = self.layer(batch[0])
            loss = self.loss(batch, output)
            return {"x": loss}

        def validation_epoch_end(self, outputs) -> None:
            pass

        def val_dataloader(self):
            dls = [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(val_dl_count)]
            return dls

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        overfit_batches=overfit_batches,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(model)
    assert trainer.num_training_batches == overfit_batches
    assert len(trainer.num_val_batches) == val_dl_count
    assert all(nbatches == overfit_batches for nbatches in trainer.num_val_batches)


@pytest.mark.parametrize("overfit_batches", [1, 2, 0.1, 0.25, 1.0])
def test_overfit_basic(tmpdir, overfit_batches):
    """Tests that only training_step can be used when overfitting."""

    model = BoringModel()
    model.validation_step = None
    total_train_samples = len(BoringModel().train_dataloader())

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, overfit_batches=overfit_batches, enable_model_summary=False
    )
    trainer.fit(model)

    assert trainer.num_val_batches == []
    assert trainer.num_training_batches == int(
        overfit_batches * (1 if isinstance(overfit_batches, int) else total_train_samples)
    )
    breakpoint()


def test_overfit_batches_raises_warning_in_case_of_sequential_sampler(tmpdir):
    class NonSequentialSampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            return iter(range(len(self.data_source)))

        def __len__(self):
            return len(self.data_source)

    class TestModel(BoringModel):
        def train_dataloader(self):
            dataset = RandomDataset(32, 64)
            sampler = NonSequentialSampler(dataset)
            return torch.utils.data.DataLoader(dataset, sampler=sampler)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, overfit_batches=2)

    with pytest.warns(UserWarning, match="requested to overfit but enabled training dataloader shuffling"):
        trainer.fit(model)

    assert isinstance(trainer.train_dataloader.loaders.sampler, SequentialSampler)
