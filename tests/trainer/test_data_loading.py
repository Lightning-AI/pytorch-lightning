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
import sys
from re import escape

import pytest
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_7
from tests.helpers import BoringModel, RandomDataset


@pytest.mark.skipif(
    sys.platform == "win32" and not _TORCH_GREATER_EQUAL_1_7, reason="Bad `torch.distributed` support on Windows"
)
@pytest.mark.parametrize("mode", (1, 2))
def test_replace_distributed_sampler(tmpdir, mode):
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

    model = TestModel(2, mode)
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir, limit_test_batches=2, plugins="ddp_find_unused_parameters_false", num_processes=1
    )
    if mode == 1:
        match = escape("missing attributes are ['num_features']")
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


def test_replace_sampler_raises():
    trainer = Trainer()
    with pytest.raises(ValueError, match="needs to subclass `torch.utils.data.DataLoader"):
        trainer.replace_sampler(object(), object(), mode="fit")  # noqa


def test_dataloaders_with_missing_keyword_arguments():
    trainer = Trainer()
    ds = RandomDataset(10, 20)

    class TestDataLoader(DataLoader):
        def __init__(self, dataset):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    match = escape("missing arguments are ['batch_sampler', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="fit")
    match = escape("missing arguments are ['batch_sampler', 'batch_size', 'drop_last', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, dataset, *args, **kwargs):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    trainer.replace_sampler(loader, sampler, mode="fit")
    trainer.replace_sampler(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, *foo, **bar):
            super().__init__(*foo, **bar)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    trainer.replace_sampler(loader, sampler, mode="fit")
    trainer.replace_sampler(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, num_feat, dataset, *args, shuffle=False):
            self.num_feat = num_feat
            super().__init__(dataset)

    loader = TestDataLoader(1, ds)
    sampler = SequentialSampler(ds)
    match = escape("missing arguments are ['batch_sampler', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="fit")
    match = escape("missing arguments are ['batch_sampler', 'batch_size', 'drop_last', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, num_feat, dataset, **kwargs):
            self.feat_num = num_feat
            super().__init__(dataset)

    loader = TestDataLoader(1, ds)
    sampler = SequentialSampler(ds)
    match = escape("missing attributes are ['num_feat']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="fit")
    match = escape("missing attributes are ['num_feat']")
    with pytest.raises(MisconfigurationException, match=match):
        trainer.replace_sampler(loader, sampler, mode="predict")


def test_replace_sampler_with_multiprocessing_context():
    """This test verifies that replace_sampler conserves multiprocessing context"""
    train = RandomDataset(32, 64)
    context = "spawn"
    train = DataLoader(train, batch_size=32, num_workers=2, multiprocessing_context=context, shuffle=True)
    trainer = Trainer()
    new_data_loader = trainer.replace_sampler(train, SequentialSampler(train.dataset))
    assert new_data_loader.multiprocessing_context == train.multiprocessing_context


def test_dataloader_reinit_for_subclass():
    class CustomDataLoader(DataLoader):
        def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            dummy_kwarg=None,
        ):
            super().__init__(
                dataset,
                batch_size,
                shuffle,
                sampler,
                batch_sampler,
                num_workers,
                collate_fn,
                pin_memory,
                drop_last,
                timeout,
                worker_init_fn,
            )
            self.dummy_kwarg = dummy_kwarg
            self.something_unrelated = 1

    trainer = Trainer(num_processes=1, accelerator="ddp_cpu")

    class CustomDummyObj:
        sampler = None

    result = trainer.auto_add_sampler(CustomDummyObj(), shuffle=True)
    assert isinstance(result, CustomDummyObj), "Wrongly reinstantiated data loader"

    dataset = list(range(10))
    result = trainer.auto_add_sampler(CustomDataLoader(dataset), shuffle=True)
    assert isinstance(result, DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert result.dummy_kwarg is None

    # Shuffled DataLoader should also work
    result = trainer.auto_add_sampler(CustomDataLoader(dataset, shuffle=True), shuffle=True)
    assert isinstance(result, DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert result.dummy_kwarg is None

    class CustomSampler(Sampler):
        pass

    # Should raise an error if existing sampler is being replaced
    dataloader = CustomDataLoader(dataset, sampler=CustomSampler(dataset))
    with pytest.raises(MisconfigurationException, match="will be replaced  by `DistributedSampler`"):
        trainer.auto_add_sampler(dataloader, shuffle=True)


def test_loader_detaching():
    """Checks that the loader has been resetted after the entrypoint"""

    class LoaderTestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert len(model.train_dataloader()) == 10
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            assert len(model.val_dataloader()) == 10
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            assert len(model.test_dataloader()) == 10
            return super().test_step(batch, batch_idx)

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            assert len(model.predict_dataloader()) == 10
            return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    loader = DataLoader(RandomDataset(32, 10), batch_size=1)

    model = LoaderTestModel()

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer = Trainer(fast_dev_run=1)
    trainer.fit(model, loader, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.validate(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.predict(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.test(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64
