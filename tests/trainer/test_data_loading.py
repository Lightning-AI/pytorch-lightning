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
from re import escape

import pytest
from torch.utils.data import DataLoader, DistributedSampler
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

    def __init__(self, numbers_test_dataloaders, mode):
        super().__init__()
        self._numbers_test_dataloaders = numbers_test_dataloaders
        self._mode = mode

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return super().test_step(batch, batch_idx)

    def on_test_start(self) -> None:
        dataloader = self.trainer.test_dataloaders[0]
        assert isinstance(dataloader, CustomDataLoader)
        assert dataloader.batch_size is None

        batch_sampler = dataloader.batch_sampler
        assert isinstance(batch_sampler, CustomBatchSampler)
        assert batch_sampler.batch_size == 1
        assert batch_sampler.drop_last
        assert isinstance(batch_sampler.sampler, DistributedSampler)

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


@pytest.mark.parametrize('mode', (1, 2))
def test_replace_distributed_sampler(tmpdir, mode):
    model = TestModel(2, mode)
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir, limit_test_batches=2, plugins="ddp_find_unused_parameters_false", num_processes=1
    )
    if mode == 1:
        match = escape("Missing arguments are ['num_features']")
        with pytest.raises(MisconfigurationException, match=match):
            trainer.test(model)
    else:
        trainer.test(model)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_dataloader_warnings(num_workers):

    class TestModel(BoringModel):

        def on_train_start(self, *_) -> None:
            raise SystemExit()

    dl = DataLoader(RandomDataset(32, 64), num_workers=num_workers)
    if hasattr(dl, "persistent_workers"):
        if num_workers == 0:
            warn_str = "Consider setting num_workers>0 and persistent_workers=True"
        else:
            warn_str = "Consider setting persistent_workers=True"
    else:
        warn_str = "Consider setting accelerator=ddp"

    trainer = Trainer(accelerator="ddp_spawn")
    with pytest.warns(UserWarning, match=warn_str), pytest.raises(SystemExit):
        trainer.fit(TestModel(), dl)


def test_dataloaders_with_missing_keyword_arguments():
    trainer = Trainer()
    ds = RandomDataset(10, 20)

    class TestDataLoader(DataLoader):

        def __init__(self, dataset):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    match = escape("['batch_sampler', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode='fit')
    match = escape("['batch_sampler', 'batch_size', 'drop_last', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode='predict')

    class TestDataLoader(DataLoader):

        def __init__(self, dataset, *args, **kwargs):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    trainer.replace_sampler(loader, sampler, mode='fit')
    trainer.replace_sampler(loader, sampler, mode='predict')

    class TestDataLoader(DataLoader):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    trainer.replace_sampler(loader, sampler, mode='fit')
    trainer.replace_sampler(loader, sampler, mode='predict')

    class TestDataLoader(DataLoader):

        def __init__(self, num_feat, dataset, *args, shuffle=False):
            self.num_feat = num_feat
            super().__init__(dataset)

    loader = TestDataLoader(1, ds)
    sampler = SequentialSampler(ds)
    match = escape("['batch_sampler', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode='fit')
    match = escape("['batch_sampler', 'batch_size', 'drop_last', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode='predict')


def test_replace_sampler_with_multiprocessing_context():
    """This test verifies that replace_sampler conserves multiprocessing context"""
    train = RandomDataset(32, 64)
    context = 'spawn'
    train = DataLoader(train, batch_size=32, num_workers=2, multiprocessing_context=context, shuffle=True)
    trainer = Trainer()
    new_data_loader = trainer.replace_sampler(train, SequentialSampler(train.dataset))
    assert new_data_loader.multiprocessing_context == train.multiprocessing_context
