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

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, RandomDataset


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


class TestModel(BoringModel):

    def __init__(self, numbers_test_dataloaders,
                 save_preds_on_dl_idx, failure):
        super().__init__()
        self._numbers_test_dataloaders = numbers_test_dataloaders
        self._save_preds_on_dl_idx = save_preds_on_dl_idx
        self._failure = failure

    def create_dataset(self):
        dataloader_cls = FailureCustomDataLoader if self._failure > 0 else CustomDataLoader
        return dataloader_cls(32, IndexedRandomDataset(32, 64), batch_size=2)

    def test_dataloader(self):
        return [self.create_dataset()] * self._numbers_test_dataloaders


def check_prediction_collection(tmpdir, save_preds_on_dl_idx, accelerator, gpus,
                                num_dl_idx, failure=0):
    num_processes = 2
    limit_test_batches = 2
    trainer_args = {
        "default_root_dir": tmpdir,
        "limit_test_batches" : limit_test_batches,
        "accelerator": accelerator,
    }

    if accelerator == "ddp_cpu":
        trainer_args["num_processes"] = num_processes
    else:
        trainer_args["gpus"] = gpus

    model = TestModel(num_dl_idx, save_preds_on_dl_idx, failure)
    model.test_epoch_end = None

    trainer = Trainer(**trainer_args)
    if failure == 1:
        match = "Missing attributes are"
    else:
        match = "DistributedSampler within"
    with pytest.raises(MisconfigurationException, match=match):
        _ = trainer.test(model)
        return


@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_misconfiguration_on_dataloader(tmpdir):
    """
    Test Lightning raise a MisConfiguration error as we can't re-instantiate user Dataloader
    """
    check_prediction_collection(tmpdir, True, "ddp", 2, 2, failure=1)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires a GPU machine")
def test_prediction_collection_1_gpu_failure(tmpdir):
    """
    Test `PredictionCollection` will raise warning as we are using an invalid custom Dataloader
    """
    check_prediction_collection(tmpdir, True, None, 1, 1, failure=2)
