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
import os

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset


class IndexedRandomDataset(RandomDataset):

    def __getitem__(self, index):
        return self.data[index]


class CustomDataLoader(DataLoader):

    def __init__(self, num_features, dataset, *args, **kwargs):
        self.num_features = num_features
        super().__init__(dataset, *args, **kwargs)


class FailureCustomDataLoader(DataLoader):

    def __init__(self, num_features, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)


class CustomBatchSampler(BatchSampler):
    pass


class TestModel(BoringModel):

    def __init__(self, numbers_test_dataloaders, save_preds_on_dl_idx, mode):
        super().__init__()
        self._numbers_test_dataloaders = numbers_test_dataloaders
        self._save_preds_on_dl_idx = save_preds_on_dl_idx
        self._mode = mode

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return super().test_step(batch, batch_idx)

    def create_dataset(self):
        dataset = IndexedRandomDataset(32, 64)
        batch_sampler = None
        batch_size = 2
        if self._mode == 2:
            batch_size = 1
            batch_sampler = CustomBatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=True)
            dataloader_cls = CustomDataLoader
        else:
            dataloader_cls = FailureCustomDataLoader
        return dataloader_cls(32, dataset, batch_size=batch_size, batch_sampler=batch_sampler)

    def test_dataloader(self):
        return [self.create_dataset()] * self._numbers_test_dataloaders


def check_replace_distrubuted_sampler(tmpdir, save_preds_on_dl_idx, accelerator, gpus, num_dl_idx, mode):
    num_processes = 2
    limit_test_batches = 2
    trainer_args = {
        "default_root_dir": tmpdir,
        "limit_test_batches": limit_test_batches,
        "accelerator": accelerator,
    }

    if accelerator == "ddp_cpu":
        trainer_args["num_processes"] = num_processes
    else:
        trainer_args["gpus"] = gpus

    model = TestModel(num_dl_idx, save_preds_on_dl_idx, mode)
    model.test_epoch_end = None

    trainer = Trainer(**trainer_args)
    if mode == 1:
        match = "DistributedSampler within"
        with pytest.raises(MisconfigurationException, match=match):
            trainer.test(model)
    else:
        trainer.test(model)


@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("mode", [1, 2])
def test_replace_distrubuted_sampler_custom_dataloader_custom_batch_sampler(tmpdir, mode):
    check_replace_distrubuted_sampler(tmpdir, True, "ddp", 2, 2, mode)
