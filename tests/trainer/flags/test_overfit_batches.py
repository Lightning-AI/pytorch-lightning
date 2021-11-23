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
from tests.base.model_template import EvalModelTemplate
from tests.helpers.boring_model import BoringModel, RandomDataset


@pytest.mark.parametrize("overfit_batches", [1, 2, 0.1, 0.25, 1.0])
def test_disable_validation_when_overfit_batches_larger_than_zero(tmpdir, overfit_batches):
    """Verify that when `overfit_batches` > 0,  there will be no validation."""

    class CurrentModel(EvalModelTemplate):

        validation_step_invoked = False
        validation_epoch_end_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

        def validation_epoch_end(self, *args, **kwargs):
            self.validation_epoch_end_invoked = True
            return super().validation_epoch_end(*args, **kwargs)

    hparams = EvalModelTemplate.get_default_hparams()
    model = CurrentModel(**hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.0,
        overfit_batches=overfit_batches,
        fast_dev_run=False,
    )

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 1
    assert not model.validation_step_invoked, "`validation_step` should not run when `overfit_batches>0`"
    assert not model.validation_epoch_end_invoked, "`validation_step` should not run when `overfit_batches>0`"


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
