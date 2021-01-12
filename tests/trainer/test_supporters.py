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

import pytest
import torch
from torch.utils.data import TensorDataset

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
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

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
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_dict_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given mapping loaders"""
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_min_size():
    """Test `CombinedLoader` of mode 'min_size' given sequence loaders"""
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given sequence loaders"""
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


class IndexedRandomDataset(RandomDataset):

    def __getitem__(self, index):
        return {"index": index, "batch": self.data[index]}


class TestModel(BoringModel):

    def __init__(self, num_test_dataloaders, save_predictions_on_dataloader_idx):
        super().__init__()
        self._num_test_dataloaders = num_test_dataloaders
        self._save_predictions_on_dataloader_idx = save_predictions_on_dataloader_idx

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        indexes, x = batch["index"], batch["batch"]
        output = self.layer(x)
        loss = self.loss(batch, output)

        if dataloader_idx is not None and dataloader_idx == 1 and not self._save_predictions_on_dataloader_idx:
            return {"y": loss}

        predictions = []
        for idx in range(len(indexes)):
            id = indexes[idx]
            prediction = output[idx]
            predictions.append({"id": id, "prediction":prediction})

        self.trainer.predictions.add(predictions)
        return {"y": loss}

    def create_dataset(self):
        return torch.utils.data.DataLoader(IndexedRandomDataset(32, 64), batch_size=2)

    def test_dataloader(self):
        return [self.create_dataset() for _ in range(self._num_test_dataloaders)]


def test_prediction_collection(tmpdir, num_test_dataloaders,
                               save_predictions_on_dataloader_idx, accelerator, gpus=None):
    model = TestModel(num_test_dataloaders, save_predictions_on_dataloader_idx)
    model.test_epoch_end = None
    limit_test_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_test_batches=limit_test_batches,
        accelerator=accelerator,
        gpus=gpus
    )
    results = trainer.test(model)
    assert len(results) == num_test_dataloaders
    for dl_idx in range(num_test_dataloaders):
        result = results[dl_idx]
        if not save_predictions_on_dataloader_idx and num_test_dataloaders == 2 and dl_idx == 1:
            assert "predictions" not in result
        else:
            assert "predictions" in result
            predictions = result["predictions"]
            assert len(predictions) == limit_test_batches * 2 * trainer.world_size


@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize('save_predictions_on_dataloader_idx', [False, True])
@pytest.mark.parametrize('num_test_dataloaders', [1, 2])
@pytest.mark.parametrize('accelerator', ["ddp"])
@pytest.mark.parametrize('gpus', [1, 2])
def test_prediction_collection_ddp(tmpdir, num_test_dataloaders, save_predictions_on_dataloader_idx, accelerator, gpus):

    """
    Test `PredictionCollection` reduce properly in ddp mode
    """
    test_prediction_collection(tmpdir, num_test_dataloaders, save_predictions_on_dataloader_idx, accelerator, gpus=gpus)
