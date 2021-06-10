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
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.supporters import (
    _nested_calc_num_data,
    CombinedDataset,
    CombinedLoader,
    CombinedLoaderIterator,
    CycleIterator,
    prefetch_iterator,
    TensorRunningAccum,
)
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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
    assert iterator.__len__() == float("inf")

    # test infinite loop
    for idx, item in enumerate(iterator):
        if idx == 1000:
            break
    assert item == 0


def test_prefetch_iterator():
    """ Test the prefetch_iterator with PyTorch IterableDataset. """

    class IterDataset(IterableDataset):

        def __iter__(self):
            yield 1
            yield 2
            yield 3

    dataset = IterDataset()
    iterator = prefetch_iterator(dataset)
    assert [item for item in iterator] == [(1, False), (2, False), (3, True)]

    class EmptyIterDataset(IterableDataset):

        def __iter__(self):
            return iter([])

    dataset = EmptyIterDataset()
    iterator = prefetch_iterator(dataset)
    assert [item for item in iterator] == []


@pytest.mark.parametrize(
    ["dataset_1", "dataset_2"],
    [
        ([list(range(10)), list(range(20))]),
        ([range(10), range(20)]),
        ([torch.randn(10, 3, 2), torch.randn(20, 5, 6)]),
        ([TensorDataset(torch.randn(10, 3, 2)),
          TensorDataset(torch.randn(20, 5, 6))]),
    ],
)
def test_combined_dataset(dataset_1, dataset_2):
    """Verify the length of the CombinedDataset"""
    datasets = [dataset_1, dataset_2]
    combined_dataset = CombinedDataset(datasets)

    assert combined_dataset.max_len == 20
    assert combined_dataset.min_len == len(combined_dataset) == 10


def test_combined_dataset_length_mode_error():
    dset = CombinedDataset([range(10)])
    with pytest.raises(MisconfigurationException, match="Invalid Mode"):
        dset._calc_num_data([range(10)], "test")


def test_combined_loader_iterator_dict_min_size():
    """Test `CombinedLoaderIterator` given mapping loaders"""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_iter = CombinedLoaderIterator(loaders)

    for idx, item in enumerate(combined_iter):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == min(len(loaders["a"]), len(loaders["b"])) - 1


def test_combined_loader_init_mode_error():
    """Test the ValueError when constructing `CombinedLoader`"""
    with pytest.raises(MisconfigurationException, match="Invalid Mode"):
        CombinedLoader([range(10)], "testtt")


def test_combined_loader_loader_type_error():
    """Test the ValueError when wrapping the loaders"""
    with pytest.raises(TypeError, match="Expected data to be int, Sequence or Mapping, but got NoneType"):
        CombinedLoader(None, "max_size_cycle")


def test_combined_loader_calc_length_mode_error():
    """Test the ValueError when calculating the number of batches"""
    with pytest.raises(TypeError, match="Expected data to be int, Sequence or Mapping, but got NoneType"):
        CombinedLoader._calc_num_batches(None)


def test_combined_loader_dict_min_size():
    """Test `CombinedLoader` of mode 'min_size' given mapping loaders"""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_loader = CombinedLoader(loaders, "min_size")

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_dict_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given mapping loaders"""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_loader = CombinedLoader(loaders, "max_size_cycle")

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_min_size():
    """Test `CombinedLoader` of mode 'min_size' given sequence loaders"""
    loaders = [
        torch.utils.data.DataLoader(range(10), batch_size=4),
        torch.utils.data.DataLoader(range(20), batch_size=5),
    ]

    combined_loader = CombinedLoader(loaders, "min_size")

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given sequence loaders"""
    loaders = [
        torch.utils.data.DataLoader(range(10), batch_size=4),
        torch.utils.data.DataLoader(range(20), batch_size=5),
    ]

    combined_loader = CombinedLoader(loaders, "max_size_cycle")

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


@pytest.mark.parametrize(
    ["input_data", "compute_func", "expected_length"],
    [
        ([*range(10), list(range(1, 20))], min, 0),
        ([*range(10), list(range(1, 20))], max, 19),
        ([*range(10), {str(i): i
                       for i in range(1, 20)}], min, 0),
        ([*range(10), {str(i): i
                       for i in range(1, 20)}], max, 19),
        ({
            **{str(i): i
               for i in range(10)}, "nested": {str(i): i
                                               for i in range(1, 20)}
        }, min, 0),
        ({
            **{str(i): i
               for i in range(10)}, "nested": {str(i): i
                                               for i in range(1, 20)}
        }, max, 19),
        ({
            **{str(i): i
               for i in range(10)}, "nested": list(range(20))
        }, min, 0),
        ({
            **{str(i): i
               for i in range(10)}, "nested": list(range(20))
        }, max, 19),
    ],
)
def test_nested_calc_num_data(input_data, compute_func, expected_length):
    calculated_length = _nested_calc_num_data(input_data, compute_func)

    assert calculated_length == expected_length


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "PL_TRAINER_GPUS": "2"})
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('torch.cuda.is_available', return_value=True)
def test_combined_data_loader_validation_test(cuda_available_mock, device_count_mock, tmpdir):
    """
    This test makes sure distributed sampler has been properly injected in dataloaders
    when using CombinedLoader
    """

    class CustomDataset(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    dataloader = CombinedLoader({
        "a": DataLoader(CustomDataset(range(10))),
        "b": {
            "c": DataLoader(CustomDataset(range(10))),
            "d": DataLoader(CustomDataset(range(10)))
        },
        "e": [DataLoader(CustomDataset(range(10))),
              DataLoader(CustomDataset(range(10)))]
    })

    trainer = Trainer(replace_sampler_ddp=True, accelerator="ddp", gpus=2)
    dataloader = trainer.auto_add_sampler(dataloader, shuffle=True)
    _count = 0

    def _assert_distributed_sampler(v):
        nonlocal _count
        _count += 1
        assert isinstance(v, DistributedSampler)

    apply_to_collection(dataloader.sampler, Sampler, _assert_distributed_sampler)
    assert _count == 5
