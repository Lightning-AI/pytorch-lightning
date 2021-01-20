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
from collections import Sequence
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.supporters import (
    CombinedDataset,
    CombinedLoader,
    CombinedLoaderIterator,
    CycleIterator,
    TensorRunningAccum,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, RandomDataset


def test_tensor_running_accum_reset():
    """ Test that reset would set all attributes to the initialization state """

    window_length = 10

    accum = TensorRunningAccum(window_length=window_length)
    assert accum.last() is None
    assert accum.mean() is None

    accum.append(torch.tensor(1.5))
    assert accum.last() == torch.tensor(1.5)
    assert accum.mean() == torch.tensor(1.5)

    accum.reset()
    assert accum.window_length == window_length
    assert accum.memory is None
    assert accum.current_idx == 0
    assert accum.last_idx is None
    assert not accum.rotated


def test_cycle_iterator():
    """Test the cycling function of `CycleIterator`"""
    iterator = CycleIterator(range(100), 1000)
    assert len(iterator) == 1000
    for idx, item in enumerate(iterator):
        assert item < 100

    assert idx == len(iterator) - 1


def test_none_length_cycle_iterator():
    """Test the infinite cycling function of `CycleIterator`"""
    iterator = CycleIterator(range(100))
    assert iterator.__len__() == float('inf')

    # test infinite loop
    for idx, item in enumerate(iterator):
        if idx == 1000:
            break
    assert item == 0


@pytest.mark.parametrize(['dataset_1', 'dataset_2'], [
    ([list(range(10)), list(range(20))]),
    ([range(10), range(20)]),
    ([torch.randn(10, 3, 2), torch.randn(20, 5, 6)]),
    ([TensorDataset(torch.randn(10, 3, 2)), TensorDataset(torch.randn(20, 5, 6))])
])
def test_combined_dataset(dataset_1, dataset_2):
    """Verify the length of the CombinedDataset"""
    datasets = [dataset_1, dataset_2]
    combined_dataset = CombinedDataset(datasets)

    assert combined_dataset.max_len == 20
    assert combined_dataset.min_len == len(combined_dataset) == 10


def test_combined_dataset_length_mode_error():
    with pytest.raises(MisconfigurationException, match='Invalid Mode'):
        CombinedDataset._calc_num_data([range(10)], 'test')


def test_combined_loader_iterator_dict_min_size():
    """Test `CombinedLoaderIterator` given mapping loaders"""
    loaders = {'a': DataLoader(range(10), batch_size=4),
               'b': DataLoader(range(20), batch_size=5)}

    combined_iter = CombinedLoaderIterator(loaders)

    for idx, item in enumerate(combined_iter):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == min(len(loaders['a']), len(loaders['b'])) - 1


def test_combined_loader_init_mode_error():
    """Test the ValueError when constructing `CombinedLoader`"""
    with pytest.raises(MisconfigurationException, match='selected unsupported mode'):
        CombinedLoader([range(10)], 'testtt')


def test_combined_loader_loader_type_error():
    """Test the ValueError when wrapping the loaders"""
    with pytest.raises(ValueError, match='Invalid Datatype'):
        CombinedLoader(None, 'max_size_cycle')


def test_combined_loader_calc_length_mode_error():
    """Test the ValueError when calculating the number of batches"""
    with pytest.raises(TypeError, match='Got Type NoneType, but expected one of Sequence, int or Mapping'):
        CombinedLoader._calc_num_batches(None)


def test_combined_loader_dict_min_size():
    """Test `CombinedLoader` of mode 'min_size' given mapping loaders"""
    loaders = {'a': DataLoader(range(10), batch_size=4),
               'b': DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_dict_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given mapping loaders"""
    loaders = {'a': DataLoader(range(10), batch_size=4),
               'b': DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_min_size():
    """Test `CombinedLoader` of mode 'min_size' given sequence loaders"""
    loaders = [DataLoader(range(10), batch_size=4),
               DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given sequence loaders"""
    loaders = [DataLoader(range(10), batch_size=4),
               DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


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
                 save_preds_on_dl_idx, failure, enable_predict_auto_id):
        super().__init__()
        self._numbers_test_dataloaders = numbers_test_dataloaders
        self._save_preds_on_dl_idx = save_preds_on_dl_idx
        self._failure = failure
        self.enable_predict_auto_id = enable_predict_auto_id

    def create_dataset(self):
        dataloader_cls = FailureCustomDataLoader if self._failure > 0 else CustomDataLoader
        return dataloader_cls(32, IndexedRandomDataset(32, 64), batch_size=2)

    def test_dataloader(self):
        return [self.create_dataset()] * self._numbers_test_dataloaders


def check_prediction_collection(tmpdir, save_preds_on_dl_idx, accelerator, gpus,
                                num_dl_idx, failure=0, enable_predict_auto_id=True):
    num_processes = 2
    limit_test_batches = 2
    trainer_args = {
        "default_root_dir": tmpdir,
        "limit_test_batches" : limit_test_batches,
        "accelerator": accelerator,
        "enable_predict_auto_id": enable_predict_auto_id
    }

    if accelerator == "ddp_cpu":
        trainer_args["num_processes"] = num_processes
        size = num_processes
    else:
        trainer_args["gpus"] = gpus
        size = gpus

    model = TestModel(num_dl_idx, save_preds_on_dl_idx, failure, enable_predict_auto_id)
    model.test_epoch_end = None

    trainer = Trainer(**trainer_args)
    if failure == 1:
        try:
            _ = trainer.test(model)
        except MisconfigurationException as e:
            assert "Missing attributes are {'num_features'}." in str(e)
            return

    else:
        try:
            _ = trainer.test(model)
        except MisconfigurationException as e:
            assert "inject DistributedSampler within FailureCustomDataLoader" in str(e)


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
