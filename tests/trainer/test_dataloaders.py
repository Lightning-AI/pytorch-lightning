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
from unittest.mock import call, Mock, patch

import numpy
import pytest
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import _auto_add_worker_init_fn, has_iterable_dataset, has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset, RandomIterableDataset, RandomIterableDatasetWithLen
from tests.helpers.dataloaders import CustomInfDataloader, CustomNotImplementedErrorDataloader
from tests.helpers.runif import RunIf


class MultiValDataLoaderBoringModel(BoringModel):
    def val_dataloader(self):
        return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64), batch_size=8)]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return super().validation_step(batch, batch_idx)

    def validation_epoch_end(self, *args, **kwargs):
        pass


class MultiTestDataLoaderBoringModel(BoringModel):
    def test_dataloader(self):
        return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64), batch_size=8)]

    def test_step(self, batch, batch_idx, dataloader_idx):
        return super().test_step(batch, batch_idx)

    def test_epoch_end(self, *args, **kwargs):
        pass


class MultiEvalDataLoaderModel(MultiValDataLoaderBoringModel, MultiTestDataLoaderBoringModel):
    pass


def test_fit_train_loader_only(tmpdir):
    model = BoringModel()
    train_dataloader = model.train_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.validation_step = None
    model.validation_epoch_end = None

    model.test_step = None
    model.test_epoch_end = None

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dataloader)


def test_fit_val_loader_only(tmpdir):
    model = BoringModel()
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.test_step = None
    model.test_epoch_end = None

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


@pytest.mark.parametrize("dataloader_options", [dict(val_check_interval=10000)])
def test_dataloader_config_errors_runtime(tmpdir, dataloader_options):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, **dataloader_options)
    with pytest.raises(ValueError, match="less than or equal to the number of the training batches"):
        trainer.fit(model)


@pytest.mark.parametrize(
    "dataloader_options",
    [
        dict(limit_train_batches=-0.1),
        dict(limit_train_batches=1.2),
        dict(limit_val_batches=-0.1),
        dict(limit_val_batches=1.2),
        dict(limit_test_batches=-0.1),
        dict(limit_test_batches=1.2),
        dict(val_check_interval=-0.1),
        dict(val_check_interval=1.2),
        dict(overfit_batches=-0.1),
        dict(overfit_batches=1.2),
    ],
)
def test_dataloader_config_errors_init(tmpdir, dataloader_options):
    with pytest.raises(MisconfigurationException, match="passed invalid value"):
        Trainer(default_root_dir=tmpdir, max_epochs=1, **dataloader_options)


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""

    model = MultiValDataLoaderBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.3, limit_train_batches=1.0)
    trainer.fit(model)

    # verify training completed
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # verify there are 2 val loaders
    assert len(trainer.val_dataloaders) == 2, "Multiple val_dataloaders not initiated properly"


@pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
def test_multiple_eval_dataloader(tmpdir, ckpt_path):
    """Verify multiple evaluation dataloaders."""
    model = MultiEvalDataLoaderModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=10, limit_train_batches=100)
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path if ckpt_path == "specific" else ckpt_path

    trainer.validate(ckpt_path=ckpt_path, verbose=False)
    # verify there are 2 loaders
    assert len(trainer.val_dataloaders) == 2

    trainer.test(ckpt_path=ckpt_path, verbose=False)
    assert len(trainer.test_dataloaders) == 2


def test_train_dataloader_passed_to_fit(tmpdir):
    """Verify that train dataloader can be passed to fit."""

    # only train passed to fit
    model = BoringModel()
    train_loader = model.train_dataloader()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)
    fit_options = dict(train_dataloaders=train_loader)
    trainer.fit(model, **fit_options)
    assert trainer.num_training_batches == 2
    assert trainer.train_dataloader.loaders == train_loader

    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
@pytest.mark.parametrize("n", (1, 2))
def test_dataloaders_passed_to_fn(tmpdir, ckpt_path, n):
    """Verify that dataloaders can be passed."""

    train_dataloaders = DataLoader(RandomDataset(32, 64))
    if n == 1:
        model = BoringModel()
        eval_dataloaders = DataLoader(RandomDataset(32, 64))
    else:
        model = MultiEvalDataLoaderModel()
        eval_dataloaders = [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    # multiple val dataloaders passed to fit
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=eval_dataloaders)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert len(trainer.val_dataloaders) == n

    if ckpt_path == "specific":
        ckpt_path = trainer.checkpoint_callback.best_model_path

    trainer.test(dataloaders=eval_dataloaders, ckpt_path=ckpt_path)
    assert len(trainer.test_dataloaders) == n

    trainer.validate(dataloaders=eval_dataloaders, ckpt_path=ckpt_path)
    assert len(trainer.val_dataloaders) == n


class DummyModel(BoringModel):
    def training_step(self, batch, batch_idx):
        self.log("loss", self.global_step)
        return super().training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.log("val_log", self.current_epoch)


class Counter(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_count = 0
        self.val_epoch_count = 0
        self.test_epoch_count = 0
        self.train_batches_seen = 0
        self.val_batches_seen = 0
        self.test_batches_seen = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batches_seen += 1

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_count += 1

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.val_batches_seen += 1

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.test_batches_seen += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_count += 1

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_epoch_count += 1


def test_inf_dataloaders_with_limit_percent_batches(tmpdir):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit in percent."""

    epoch_cb = Counter()
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        max_epochs=1,
        callbacks=[epoch_cb],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
    )
    model = DummyModel()

    batch_size = 8
    train_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)
    val_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)
    test_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)

    num_batches = 128 / batch_size
    for dl in (train_dl, val_dl, test_dl):
        if has_len_all_ranks(dl, trainer.strategy, model):
            assert len(dl) == num_batches
        else:
            assert sum(1 for _ in dl) == num_batches

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.num_training_batches == float("inf")
    assert epoch_cb.train_epoch_count == 1

    assert trainer.num_val_batches[0] == float("inf")
    assert epoch_cb.val_epoch_count == 1

    trainer.test(model, dataloaders=test_dl)
    assert trainer.num_test_batches[0] == float("inf")
    assert epoch_cb.test_epoch_count == 1


@pytest.mark.parametrize(
    ["dataset", "limit_train_batches"],
    [
        (RandomDataset(32, 128), 10),
        (RandomIterableDataset(32, 128), 10),
        (RandomIterableDatasetWithLen(32, 128), 10),
    ],
)
def test_dataloaders_with_limit_train_batches(tmpdir, dataset, limit_train_batches):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""

    epoch_cb = Counter()
    max_epochs = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[epoch_cb],
        limit_train_batches=limit_train_batches,
    )
    model = DummyModel()

    batch_size = 8
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = DataLoader(dataset=dataset, batch_size=batch_size)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.num_training_batches == limit_train_batches
    assert epoch_cb.train_epoch_count == max_epochs
    assert epoch_cb.train_batches_seen == limit_train_batches * max_epochs


@pytest.mark.parametrize(
    "dataset",
    [
        RandomDataset(32, 128),
        RandomIterableDataset(32, 128),
        RandomIterableDatasetWithLen(32, 128),
    ],
)
def test_dataloaders_with_limit_val_batches(tmpdir, dataset):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""

    epoch_cb = Counter()
    callbacks = [epoch_cb]
    enable_checkpointing = False

    max_epochs = 2
    limit_val_batches = 10
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=callbacks,
        limit_val_batches=limit_val_batches,
        enable_checkpointing=enable_checkpointing,
    )
    model = DummyModel()

    batch_size = 8
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = DataLoader(dataset=dataset, batch_size=batch_size)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.num_val_batches[0] == limit_val_batches
    assert epoch_cb.val_epoch_count == max_epochs
    assert epoch_cb.val_batches_seen == limit_val_batches * max_epochs


@pytest.mark.skip()
@pytest.mark.parametrize(
    "dataset",
    [
        RandomDataset(32, 128),
        RandomIterableDataset(32, 128),
        RandomIterableDatasetWithLen(32, 128),
    ],
)
def test_datasets_dataloaders_with_limit_num_batches(tmpdir, dataset):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""

    epoch_cb = Counter()
    max_epochs = 2
    limit_batches = 10
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[epoch_cb],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
    )
    model = DummyModel()

    batch_size = 8
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = DataLoader(dataset=dataset, batch_size=batch_size)
    test_dl = DataLoader(dataset=dataset, batch_size=batch_size)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.num_training_batches == limit_batches
    assert trainer.num_val_batches[0] == limit_batches
    assert epoch_cb.train_epoch_count == max_epochs
    assert epoch_cb.train_batches_seen == limit_batches * max_epochs
    assert epoch_cb.val_epoch_count == max_epochs
    assert epoch_cb.val_batches_seen == limit_batches * max_epochs

    trainer.test(model, dataloaders=test_dl)
    assert trainer.num_test_batches[0] == limit_batches
    assert epoch_cb.test_epoch_count == 1


@pytest.mark.skip()
@pytest.mark.parametrize(
    ["limit_train_batches", "limit_val_batches", "limit_test_batches"],
    [(1.0, 1.0, 1.0), (0.2, 0.4, 0.4)],
)
def test_dataloaders_with_limit_percent_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify num_batches for train, val & test dataloaders passed with batch limit in percent."""
    model = MultiEvalDataLoaderModel()
    # train, multiple val and multiple test passed with percent_check
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )
    trainer.fit(model)
    expected_train_batches = int(len(trainer.train_dataloader) * limit_train_batches)
    expected_val_batches = [int(len(dataloader) * limit_val_batches) for dataloader in trainer.val_dataloaders]
    assert trainer.num_training_batches == expected_train_batches
    assert trainer.num_val_batches == expected_val_batches

    trainer.test(model)
    expected_test_batches = [int(len(dataloader) * limit_test_batches) for dataloader in trainer.test_dataloaders]
    assert trainer.num_test_batches == expected_test_batches


@pytest.mark.parametrize(["limit_train_batches", "limit_val_batches", "limit_test_batches"], [(1, 2, 3), (1, 2, 1e50)])
def test_dataloaders_with_limit_num_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify num_batches for train, val & test dataloaders passed with batch limit as number."""

    model = MultiEvalDataLoaderModel()

    # train, multiple val and multiple test passed with percent_check
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=0,
    )

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop.epoch_loop,
        "_evaluation_step",
        wraps=trainer.fit_loop.epoch_loop.val_loop.epoch_loop._evaluation_step,
    ) as mocked:
        trainer.fit(model)
        assert trainer.num_training_batches == limit_train_batches
        assert trainer.num_val_batches == [limit_val_batches] * len(trainer.val_dataloaders)
        assert mocked.call_count == limit_val_batches * len(trainer.val_dataloaders)

    with patch.object(
        trainer.test_loop.epoch_loop,
        "_evaluation_step",
        wraps=trainer.test_loop.epoch_loop._evaluation_step,
    ) as mocked:
        trainer.test(model)
        test_dataloader_lengths = [len(x) for x in model.test_dataloader()]
        if limit_test_batches > 1e10:
            # when the limit is greater than the number of test batches it should be the num in loaders
            assert trainer.num_test_batches == test_dataloader_lengths
            assert mocked.call_count == sum(test_dataloader_lengths)
        else:
            assert trainer.num_test_batches == [limit_test_batches] * len(trainer.test_dataloaders)
            assert mocked.call_count == limit_test_batches * len(trainer.test_dataloaders)


@pytest.mark.parametrize("fast_dev_run", [True, 1, 3, -1])
def test_dataloaders_with_fast_dev_run(tmpdir, fast_dev_run):
    """Verify num_batches for train, val & test dataloaders passed with fast_dev_run."""
    model = MultiEvalDataLoaderModel()
    trainer_options = dict(default_root_dir=tmpdir, max_epochs=2, fast_dev_run=fast_dev_run)

    if fast_dev_run == -1:
        with pytest.raises(MisconfigurationException, match="should be >= 0"):
            Trainer(**trainer_options)
    else:
        trainer = Trainer(**trainer_options)

        # fast_dev_run is set to True when it is 1
        if fast_dev_run == 1:
            fast_dev_run = True

        assert trainer.fast_dev_run is fast_dev_run

        if fast_dev_run is True:
            fast_dev_run = 1

        assert trainer.limit_train_batches == fast_dev_run
        assert trainer.limit_val_batches == fast_dev_run
        assert trainer.limit_test_batches == fast_dev_run
        assert trainer.num_sanity_val_steps == 0
        assert trainer.max_epochs == 1

        trainer.fit(model)
        assert trainer.enable_validation
        assert trainer.num_training_batches == fast_dev_run
        assert trainer.num_val_batches == [fast_dev_run] * len(trainer.val_dataloaders)

        trainer.test(model)
        assert trainer.num_test_batches == [fast_dev_run] * len(trainer.test_dataloaders)


@pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
def test_mixing_of_dataloader_options(tmpdir, ckpt_path):
    """Verify that dataloaders can be passed to fit."""

    model = BoringModel()
    eval_dataloader = DataLoader(RandomDataset(32, 64))
    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, val_dataloaders=eval_dataloader)
    assert len(trainer.val_dataloaders) == 1
    assert trainer.val_dataloaders[0] == eval_dataloader

    ckpt_path = trainer.checkpoint_callback.best_model_path if ckpt_path == "specific" else ckpt_path
    trainer.test(dataloaders=eval_dataloader, ckpt_path=ckpt_path)
    assert len(trainer.test_dataloaders) == 1
    assert trainer.test_dataloaders[0] == eval_dataloader
    assert len(trainer.test_dataloaders) == 1


def test_warning_on_zero_len_dataloader(tmpdir):
    """Test that a warning is raised if a zero-length dataloader is defined."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
    )
    dataloader = DataLoader(RandomDataset(32, 0))
    with pytest.warns(UserWarning, match="returned 0 length"):
        trainer.fit(model, dataloader)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("ckpt_path", (None, "best", "specific"))
@pytest.mark.parametrize("stage", ("train", "test", "val"))
@patch("pytorch_lightning.trainer.connectors.data_connector.multiprocessing.cpu_count", return_value=4)
def test_warning_with_few_workers(_, tmpdir, ckpt_path, stage):
    """Test that error is raised if dataloader with only a few workers is used."""

    model = BoringModel()

    train_dl = model.train_dataloader()
    train_dl.num_workers = 0

    val_dl = model.val_dataloader()
    val_dl.num_workers = 0

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)

    with pytest.warns(
        UserWarning,
        match=f'The dataloader, {stage}_dataloader{" 0" if stage != "train" else ""}, does not have many workers',
    ):
        if stage == "test":
            if ckpt_path in ("specific", "best"):
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            ckpt_path = trainer.checkpoint_callback.best_model_path if ckpt_path == "specific" else ckpt_path
            trainer.test(model, dataloaders=train_dl, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("ckpt_path", (None, "best", "specific"))
@pytest.mark.parametrize("stage", ("train", "test", "val"))
@patch("pytorch_lightning.trainer.connectors.data_connector.multiprocessing.cpu_count", return_value=4)
def test_warning_with_few_workers_multi_loader(_, tmpdir, ckpt_path, stage):
    """Test that error is raised if dataloader with only a few workers is used."""

    class CustomModel(MultiEvalDataLoaderModel):
        def training_step(self, batch, batch_idx):
            return super().training_step(batch["a_b"][0], batch_idx)

    model = CustomModel()
    val_dl = DataLoader(RandomDataset(32, 64))
    val_dl.num_workers = 0

    train_dl = DataLoader(RandomDataset(32, 64))
    train_dl.num_workers = 0

    train_multi_dl = {"a_b": [train_dl, train_dl], "c_d_e": [train_dl, train_dl, train_dl]}
    val_multi_dl = [val_dl, val_dl]
    test_multi_dl = [train_dl, train_dl]

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)

    with pytest.warns(
        UserWarning,
        match=f'The dataloader, {stage}_dataloader{" 0" if stage != "train" else ""}, does not have many workers',
    ):
        if stage == "test":
            if ckpt_path in ("specific", "best"):
                trainer.fit(model, train_dataloaders=train_multi_dl, val_dataloaders=val_multi_dl)
            ckpt_path = trainer.checkpoint_callback.best_model_path if ckpt_path == "specific" else ckpt_path
            trainer.test(model, dataloaders=test_multi_dl, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_dataloaders=train_multi_dl, val_dataloaders=val_multi_dl)


class NumpyRandomDataset(Dataset):
    # this dataset uses numpy instead of torch to produce random numbers
    size = 16

    def __getitem__(self, index):
        return numpy.random.randint(0, 100, 3)

    def __len__(self):
        return self.size


def _user_worker_init_fn(_):
    pass


@RunIf(max_torch="1.8.9")
def test_missing_worker_init_fn():
    """Test that naive worker seed initialization leads to undesired random state in subprocesses.

    PyTorch 1.9+ does not have this issue.
    """
    dataset = NumpyRandomDataset()

    seed_everything(0)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    batches0 = torch.cat(list(dataloader))

    seed_everything(0)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    batches1 = torch.cat(list(dataloader))

    is_duplicated = len(torch.unique(batches1, dim=0)) < len(dataset)
    is_deterministic = torch.eq(batches0, batches1).all()

    # depending on the OS, we either have
    # 1) the same seed in all worker processes, producing duplicate samples / augmentations, or
    # 2) different seeds in each worker process, but they are not derived from the seed of the main process
    assert not is_deterministic or is_duplicated


def test_auto_add_worker_init_fn():
    """Test Trainer adds a default worker_init_fn to the dataloader when seed_everything() is used."""
    dataset = Mock()
    dataloader = DataLoader(dataset)

    # without pl.seed_everything()
    _auto_add_worker_init_fn(dataloader, 0)
    assert dataloader.worker_init_fn is None

    # with forcefully avoiding it
    seed_everything(0, workers=False)
    _auto_add_worker_init_fn(dataloader, 0)
    assert dataloader.worker_init_fn is None

    # when user already has a worker_init_fn
    user_function = _user_worker_init_fn
    dataloader.worker_init_fn = user_function
    _auto_add_worker_init_fn(dataloader, 0)
    assert dataloader.worker_init_fn is user_function
    dataloader.worker_init_fn = None

    # main use case
    seed_everything(0, workers=True)
    _auto_add_worker_init_fn(dataloader, 0)
    assert dataloader.worker_init_fn is not None


class MultiProcessModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.batches_seen = []

    def training_step(self, batch, batch_idx):
        self.batches_seen.append(batch)

    def training_epoch_end(self, outputs):
        world_size = 2
        num_samples = NumpyRandomDataset.size
        all_batches = torch.cat(self.batches_seen)
        all_batches = self.all_gather(all_batches)
        assert all_batches.shape[0] == world_size
        all_batches = all_batches.view(-1, 3)
        assert len(torch.unique(all_batches, dim=0)) == num_samples


@RunIf(min_gpus=2)
def test_auto_add_worker_init_fn_distributed(tmpdir, monkeypatch):
    """Test that the lightning worker_init_fn takes care of dataloaders in multi-gpu/multi-node training."""
    dataset = NumpyRandomDataset()
    num_workers = 2
    batch_size = 2

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    seed_everything(0, workers=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp_spawn")
    model = MultiProcessModel()
    model.val_dataloader = None
    trainer.fit(model, train_dataloaders=dataloader)


def test_warning_with_small_dataloader_and_logging_interval(tmpdir):
    """Test that a warning message is shown if the dataloader length is too short for the chosen logging
    interval."""
    model = BoringModel()
    dataloader = DataLoader(RandomDataset(32, length=10))
    model.train_dataloader = lambda: dataloader

    with pytest.warns(UserWarning, match=r"The number of training samples \(10\) is smaller than the logging interval"):
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, log_every_n_steps=11)
        trainer.fit(model)

    with pytest.warns(UserWarning, match=r"The number of training samples \(1\) is smaller than the logging interval"):
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, log_every_n_steps=2, limit_train_batches=1)
        trainer.fit(model)


def test_warning_with_iterable_dataset_and_len(tmpdir):
    """Tests that a warning message is shown when an IterableDataset defines `__len__`."""
    model = BoringModel()
    original_dataset = model.train_dataloader().dataset

    class IterableWithoutLen(IterableDataset):
        def __iter__(self):
            return iter(original_dataset)

    class IterableWithLen(IterableWithoutLen):
        def __len__(self):
            return len(original_dataset)

    # with __len__ defined
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    dataloader = DataLoader(IterableWithLen(), batch_size=16)
    assert has_len_all_ranks(dataloader, trainer.strategy, model)
    assert has_iterable_dataset(dataloader)
    with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
        trainer.validate(model, dataloaders=[dataloader])
    with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=[dataloader])
    with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
        trainer.test(model, dataloaders=[dataloader])
    with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
        trainer.predict(model, dataloaders=[dataloader])

    # without __len__ defined
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    dataloader = DataLoader(IterableWithoutLen(), batch_size=16)
    assert not has_len_all_ranks(dataloader, trainer.strategy, model)
    assert has_iterable_dataset(dataloader)
    trainer.validate(model, dataloaders=dataloader)
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=[dataloader])
    trainer.test(model, dataloaders=dataloader)
    trainer.predict(model, dataloaders=dataloader)


@pytest.mark.parametrize("yield_at_all", (False, True))
def test_iterable_dataset_stop_iteration_at_epoch_beginning(yield_at_all):
    """Test that the training loop skips execution if the iterator is empty from the start."""

    class TestDataset(IterableDataset):
        def __init__(self, gen):
            self.gen = gen

        def __iter__(self):
            return iter(self.gen())

    class TestModel(BoringModel):
        def gen(self):
            # produce data in epoch 0, no data otherwise
            if yield_at_all and self.current_epoch == 0:
                yield torch.rand(32)
                yield torch.rand(32)
                yield torch.rand(32)

    model = TestModel()
    train_dataloader = DataLoader(TestDataset(model.gen), batch_size=2)
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=2,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    assert trainer.global_step == 2 * yield_at_all
    # even though the generator might not yield any data, the fit_loop still advances so the
    # current epoch gets increased
    assert trainer.current_epoch == 2


class DistribSamplerCallback(Callback):
    def __init__(self, expected_seeds=(0, 0, 0)):
        self.expected_seed = expected_seeds

    def on_train_start(self, trainer, pl_module):
        train_sampler = trainer.train_dataloader.sampler
        assert isinstance(train_sampler, DistributedSampler)
        assert train_sampler.shuffle
        assert train_sampler.seed == self.expected_seed[0]

    def on_validation_start(self, trainer, pl_module):
        val_sampler = trainer.val_dataloaders[0].sampler
        assert isinstance(val_sampler, DistributedSampler)
        assert not val_sampler.shuffle
        assert val_sampler.seed == self.expected_seed[1]

    def on_test_start(self, trainer, pl_module):
        test_sampler = trainer.test_dataloaders[0].sampler
        assert isinstance(test_sampler, DistributedSampler)
        assert not test_sampler.shuffle
        assert test_sampler.seed == self.expected_seed[2]


@RunIf(min_gpus=2, skip_windows=True)
def test_dataloader_distributed_sampler(tmpdir):
    """Test DistributedSampler and it's arguments for DDP backend."""
    seed_everything(123)
    model = BoringModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1],
        num_nodes=1,
        strategy="ddp_spawn",
        default_root_dir=tmpdir,
        max_steps=1,
        callbacks=[DistribSamplerCallback(expected_seeds=(123, 123, 123))],
    )
    trainer.fit(model)
    trainer.test(model)


class ModelWithDataLoaderDistributedSampler(BoringModel):
    def train_dataloader(self):
        dataloader = super().train_dataloader()
        dist_sampler = DistributedSampler(dataloader.dataset, shuffle=True, seed=11)
        return DataLoader(dataloader.dataset, batch_size=32, drop_last=False, sampler=dist_sampler, shuffle=False)


@RunIf(min_gpus=2, skip_windows=True)
def test_dataloader_distributed_sampler_already_attached(tmpdir):
    """Test DistributedSampler and it's arguments for DDP backend when DistSampler already included on
    dataloader."""
    seed_everything(123)
    model = ModelWithDataLoaderDistributedSampler()
    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1],
        num_nodes=1,
        strategy="ddp_spawn",
        default_root_dir=tmpdir,
        max_steps=100,
        callbacks=[DistribSamplerCallback(expected_seeds=(11, 123, 0))],
        replace_sampler_ddp=True,
    )
    trainer.fit(model)
    assert trainer.state.finished, "DDP Training failed"


@RunIf(min_gpus=3)
def test_batch_size_smaller_than_num_gpus(tmpdir):
    # we need at least 3 gpus for this test
    num_gpus = 3
    batch_size = 3

    class CurrentTestModel(BoringModel):
        def __init__(self, batch_size) -> None:
            super().__init__()
            self.save_hyperparameters()
            # batch norm doesn't work with batch size 1, we replace it
            self.c_d1_bn = torch.nn.ReLU()

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            loss = output["loss"]
            # we make sure to add some metrics to the output dict,
            # this is essential for this test
            output["progress_bar"] = {"train_loss": loss}
            return output

        def train_dataloader(self):
            dataset = RandomDataset(32, 64)
            # construct a dataset with a size that is not divisible by num_gpus
            # therefore the last batch will have a size < num_gpus
            size = num_gpus * self.hparams.batch_size + (num_gpus - 1)
            dataset = Subset(dataset, range(size))
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, drop_last=False)
            return dataloader

    model = CurrentTestModel(batch_size=batch_size)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0,
        accelerator="gpu",
        devices=num_gpus,
    )

    # we expect the reduction for the metrics also to happen on the last batch
    # where we will get fewer metrics than gpus
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    ["multiple_trainloader_mode", "num_training_batches"],
    [("min_size", 16), ("max_size_cycle", 64)],
)
def test_fit_multiple_train_loaders(tmpdir, multiple_trainloader_mode, num_training_batches):
    """Integration test for multiple train loaders."""

    class CustomBoringModel(BoringModel):
        def train_dataloader(self):
            loaders_a_b = [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 16))]
            loaders_c_d_e = [
                DataLoader(RandomDataset(32, 16)),
                DataLoader(RandomDataset(32, 16)),
                DataLoader(RandomDataset(32, 16)),
            ]
            # Dict[str, List[DataLoader]]
            loaders = {"a_b": loaders_a_b, "c_d_e": loaders_c_d_e}
            return loaders

        def training_step(self, batch, batch_idx):
            assert len(batch) == 2
            assert len(batch["a_b"]) == 2
            assert len(batch["c_d_e"]) == 3
            return super().training_step(batch["a_b"][0], batch_idx)

    model = CustomBoringModel()
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, multiple_trainloader_mode=multiple_trainloader_mode)
    trainer.fit(model)
    # verify the num_training_batches according to the multiple_trainloader_mode
    assert num_training_batches == trainer.num_training_batches


@pytest.mark.parametrize("check_interval", [50, 1.0])
@pytest.mark.parametrize("dataloader_wrapper", [CustomNotImplementedErrorDataloader, CustomInfDataloader])
def test_train_dataloader_not_implemented_error(tmpdir, check_interval, dataloader_wrapper):
    """Test not_implemented_error train data loader (e.g. IterableDataset)"""

    class CustomBoringModel(BoringModel):
        def train_dataloader(self):
            return dataloader_wrapper(DataLoader(RandomDataset(32, 64)))

        def val_dataloader(self):
            return dataloader_wrapper(DataLoader(RandomDataset(32, 64)))

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=5, max_epochs=1, val_check_interval=check_interval)
    trainer.fit(model)
    # verify training completed
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    "stage", [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
)
@pytest.mark.parametrize("dataloader_wrapper", [CustomNotImplementedErrorDataloader, CustomInfDataloader])
def test_inf_dataloader_raise_error_with_partial_batch_limits(tmpdir, stage, dataloader_wrapper):
    """Test limit_batch error with inf dataloader (e.g. IterableDataset)"""
    model = BoringModel()
    setattr(
        model, f"{stage.dataloader_prefix}_dataloader", lambda: dataloader_wrapper(DataLoader(RandomDataset(32, 64)))
    )
    trainer_kwargs = {"default_root_dir": tmpdir, "max_epochs": 1, f"limit_{stage.dataloader_prefix}_batches": 0.5}
    trainer = Trainer(**trainer_kwargs)
    trainer_fn = "fit" if stage == RunningStage.TRAINING else stage.value

    with pytest.raises(MisconfigurationException, match=r"using an IterableDataset .* must be `1.0` or an int"):
        getattr(trainer, trainer_fn)(model)


def test_dataloaders_load_only_once(tmpdir):
    model = BoringModel()
    tracker = Mock()

    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=0.3, limit_val_batches=0.3, max_epochs=3)
    trainer.fit(model)

    model.train_dataloader.assert_called_once()
    model.val_dataloader.assert_called_once()
    model.test_dataloader.assert_not_called()

    assert tracker.mock_calls == [call.val_dataloader(), call.train_dataloader()]


def test_dataloaders_load_only_once_no_sanity_check(tmpdir):
    model = BoringModel()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir, limit_train_batches=0.3, limit_val_batches=0.3, num_sanity_val_steps=0, max_epochs=3
    )

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer.fit(model)

    # verify the sequence
    expected_sequence = [call.train_dataloader(), call.val_dataloader()]
    assert tracker.mock_calls == expected_sequence


@pytest.mark.parametrize("n", [1, 2])
def test_dataloaders_load_every_n_epochs(tmpdir, n):
    train_reload_epochs, val_reload_epochs = [], []

    class TestModel(BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        reload_dataloaders_every_n_epochs=n,
        max_epochs=5,
    )

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer.fit(model)
    trainer.test(model)

    # Verify the sequence
    expected_sequence = [call.val_dataloader(), call.train_dataloader()]  # Sanity check first
    if n == 1:
        expected_sequence += [call.train_dataloader(), call.val_dataloader()] * 4
    elif n == 2:
        expected_sequence += [call.train_dataloader(), call.val_dataloader()] * 2
    expected_sequence += [call.test_dataloader()]

    assert tracker.mock_calls == expected_sequence

    # Verify epoch of reloads
    if n == 1:
        assert train_reload_epochs == [0, 1, 2, 3, 4]
        assert val_reload_epochs == [0, 1, 2, 3, 4]
    elif n == 2:
        assert train_reload_epochs == [0, 2, 4]
        assert val_reload_epochs == [0, 2, 4]


@pytest.mark.parametrize(
    "n, train_reload_epochs_expect, val_reload_epochs_expect",
    [
        # Sanity check at epoch 0 creates a validation dataloader, but validation is
        # checked (and in this case reloaded) every n epochs starting from epoch n-1
        (3, [0, 2, 4, 6, 8], [0, 2, 5, 8]),
        (5, [0, 2, 4, 6, 8], [0, 4, 9]),
    ],
)
def test_dataloaders_load_every_n_epochs_infrequent_val(
    tmpdir, n, train_reload_epochs_expect, val_reload_epochs_expect
):
    """Test dataloader reload behavior when infrequently checking validation set (via check_val_every_n_epoch)"""
    train_reload_epochs, val_reload_epochs = [], []

    class TestModel(BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        check_val_every_n_epoch=n,
        reload_dataloaders_every_n_epochs=2,
        max_epochs=10,
    )
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    trainer.fit(model)
    trainer.test(model)

    # Verify epoch of reloads
    assert train_reload_epochs == train_reload_epochs_expect
    assert val_reload_epochs == val_reload_epochs_expect

    model.test_dataloader.assert_called_once()


def test_dataloaders_load_every_n_epochs_frequent_val(tmpdir):
    """Test dataloader reload behavior when frequently checking validation set (via val_check_interval)"""
    train_reload_epochs, val_reload_epochs, val_check_epochs = [], [], []

    class TestModel(BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

        def validation_epoch_end(self, outputs):
            val_check_epochs.append(self.current_epoch)
            return super().validation_epoch_end(outputs)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        val_check_interval=0.3,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=3,
    )

    model.test_dataloader = Mock(wraps=model.test_dataloader)

    trainer.fit(model)
    trainer.test(model)

    # Verify epoch of reloads
    assert train_reload_epochs == [0, 1, 2]
    assert val_reload_epochs == [0, 1, 2]
    model.test_dataloader.assert_called_once()

    # Verify validation happens 3 times per epoch + 1 for sanity check
    assert val_check_epochs == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("n", ["test", -1])
def test_dataloaders_load_every_n_epochs_exception(tmpdir, n):

    with pytest.raises(MisconfigurationException, match="should be an int >"):
        Trainer(default_root_dir=tmpdir, reload_dataloaders_every_n_epochs=n)


def test_dataloaders_load_every_epoch_no_sanity_check(tmpdir):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("dummy_val", 5.0)
            return super().validation_step(batch, batch_idx)

    model = TestModel()

    # This callback tests that the evaluation metrics are available by the time we run checkpointing
    checkpoint_callback = ModelCheckpoint(monitor="dummy_val", save_top_k=1)

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=3,
        callbacks=[checkpoint_callback],
    )

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer.fit(model)
    trainer.test(model)

    expected_calls = [
        call.train_dataloader(),
        call.val_dataloader(),
        call.train_dataloader(),
        call.val_dataloader(),
        call.train_dataloader(),
        call.val_dataloader(),
        call.test_dataloader(),
    ]
    assert tracker.mock_calls == expected_calls


def test_dataloaders_load_only_once_passed_loaders(tmpdir):
    model = BoringModel()
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()
    test_dataloader = model.test_dataloader()

    # delete dataloader methods on the model
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=0.3, limit_val_batches=0.3, max_epochs=3)

    trainer.reset_train_dataloader = Mock(wraps=trainer.reset_train_dataloader)
    trainer.reset_val_dataloader = Mock(wraps=trainer.reset_val_dataloader)
    trainer.reset_test_dataloader = Mock(wraps=trainer.reset_test_dataloader)

    tracker = Mock()
    tracker.attach_mock(trainer.reset_train_dataloader, "reset_train_dataloader")
    tracker.attach_mock(trainer.reset_val_dataloader, "reset_val_dataloader")
    tracker.attach_mock(trainer.reset_test_dataloader, "reset_test_dataloader")

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

    trainer.reset_train_dataloader.assert_called_once()
    trainer.reset_val_dataloader.assert_called_once()
    trainer.reset_test_dataloader.assert_called_once()

    assert tracker.mock_calls == [
        call.reset_val_dataloader(),
        call.reset_train_dataloader(model=model),
        call.reset_test_dataloader(),
    ]


def test_dataloaders_reset_and_attach(tmpdir):
    """Test that repeated calls to Trainer.{fit,validate,test,predict} properly reset dataloaders before attaching
    the new one."""
    # the assertions compare the datasets and not dataloaders since we patch and replace the samplers
    dataloader_0 = DataLoader(dataset=RandomDataset(32, 64))
    dataloader_1 = DataLoader(dataset=RandomDataset(32, 64))
    dataloader_2 = DataLoader(dataset=RandomDataset(32, 64))
    dataloader_3 = DataLoader(dataset=RandomDataset(32, 64))
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
    )

    # 1st fit
    trainer.fit(model, train_dataloaders=dataloader_0, val_dataloaders=dataloader_1)
    assert trainer.train_dataloader.loaders.dataset is dataloader_0.dataset
    assert trainer.val_dataloaders[0].dataset is dataloader_1.dataset
    # 2nd fit
    trainer.fit_loop.max_steps += 1
    trainer.fit(model, train_dataloaders=dataloader_2, val_dataloaders=dataloader_3)
    assert trainer.train_dataloader.loaders.dataset is dataloader_2.dataset
    assert trainer.val_dataloaders[0].dataset is dataloader_3.dataset

    # 1st validate
    trainer.validate(model, dataloaders=dataloader_0)
    assert trainer.val_dataloaders[0].dataset is dataloader_0.dataset
    # 2nd validate
    trainer.validate(model, dataloaders=dataloader_1)
    assert trainer.val_dataloaders[0].dataset is dataloader_1.dataset

    # 1st test
    trainer.test(model, dataloaders=dataloader_0)
    assert trainer.test_dataloaders[0].dataset is dataloader_0.dataset
    # 2nd test
    trainer.test(model, dataloaders=dataloader_1)
    assert trainer.test_dataloaders[0].dataset is dataloader_1.dataset

    # 1st predict
    trainer.predict(model, dataloaders=dataloader_0)
    assert trainer.predict_dataloaders[0].dataset is dataloader_0.dataset
    # 2nd predict
    trainer.predict(model, dataloaders=dataloader_1)
    assert trainer.predict_dataloaders[0].dataset is dataloader_1.dataset


@pytest.mark.parametrize("multiple_trainloader_mode", ["min_size", "max_size_cycle"])
def test_correct_dataloader_idx_in_hooks(tmpdir, multiple_trainloader_mode):
    """Check the correct dataloader_idx inside hooks."""

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.val_call_count = 0
            self.test_call_count = 0

        def assert_dataloader_idx_hook(self, dataloader_idx):
            if self.trainer.training:
                assert dataloader_idx == 0
            elif self.trainer.validating:
                assert dataloader_idx == (0 if self.val_call_count <= 5 else 1)
            elif self.trainer.testing:
                assert dataloader_idx == (0 if self.test_call_count <= 5 else 1)

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            self.assert_dataloader_idx_hook(dataloader_idx)
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        def on_before_batch_transfer(self, batch, dataloader_idx):
            # incrementing here since this is the first hook called at each step
            if self.trainer.validating:
                self.val_call_count += 1
            elif self.trainer.testing:
                self.test_call_count += 1

            self.assert_dataloader_idx_hook(dataloader_idx)
            return super().on_before_batch_transfer(batch, dataloader_idx)

        def on_after_batch_transfer(self, batch, dataloader_idx):
            self.assert_dataloader_idx_hook(dataloader_idx)
            return super().on_after_batch_transfer(batch, dataloader_idx)

        def training_step(self, batch, batch_idx):
            return super().training_step(batch["a"], batch_idx)

        def validation_step(self, batch, batch_idx, dataloader_idx):
            self.assert_dataloader_idx_hook(dataloader_idx)
            out = super().validation_step(batch, batch_idx)
            loss = out.pop("x")
            out[f"val_loss_{dataloader_idx}"] = loss
            return out

        def test_step(self, batch, batch_idx, dataloader_idx):
            self.assert_dataloader_idx_hook(dataloader_idx)
            out = super().test_step(batch, batch_idx)
            loss = out.pop("y")
            out[f"test_loss_{dataloader_idx}"] = loss
            return out

        def predict(self, batch, batch_idx, dataloader_idx):
            self.assert_dataloader_idx_hook(dataloader_idx)
            return super().predict(batch, batch_idx, dataloader_idx)

        def assert_epoch_end_outputs(self, outputs, mode):
            assert len(outputs) == 2
            assert all(f"{mode}_loss_0" in x for x in outputs[0])
            assert all(f"{mode}_loss_1" in x for x in outputs[1])

        def validation_epoch_end(self, outputs):
            self.assert_epoch_end_outputs(outputs, mode="val")

        def test_epoch_end(self, outputs):
            self.assert_epoch_end_outputs(outputs, mode="test")

        def train_dataloader(self):
            return {"a": DataLoader(RandomDataset(32, 64)), "b": DataLoader(RandomDataset(32, 64))}

        def val_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

        def test_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

        def predict_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = CustomBoringModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=5, multiple_trainloader_mode=multiple_trainloader_mode)

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    trainer.test(model)

    preds = trainer.predict(model)
    assert len(preds) == 2
    assert all(len(x) == 5 for x in preds)


def test_request_dataloader(tmpdir):
    """This test asserts dataloader can be wrapped."""

    class DataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
            self._iter = iter(self.loader)

        def __iter__(self):
            self._iter = iter(self.loader)
            return self._iter

        def __next__(self):
            return next(self._iter)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.on_train_batch_start_called = False
            self.on_val_batch_start_called = False

        def train_dataloader(self):
            loader = super().train_dataloader()
            return DataLoaderWrapper(loader)

        def on_train_batch_start(self, batch, batch_idx: int) -> None:
            assert isinstance(self.trainer.train_dataloader.loaders, DataLoaderWrapper)
            self.on_train_batch_start_called = True

        def val_dataloader(self):
            loader = super().val_dataloader()
            return DataLoaderWrapper(loader)

        def on_validation_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert isinstance(self.trainer.val_dataloaders[0], DataLoaderWrapper)
            self.on_val_batch_start_called = True

    trainer = Trainer(
        default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, max_epochs=1
    )
    model = TestModel()
    trainer.fit(model)
    trainer.test(model)
    assert model.on_train_batch_start_called
    assert model.on_val_batch_start_called


@pytest.mark.parametrize("num_loaders", [1, 2])
def test_multiple_dataloaders_with_random_sampler_overfit_batches(num_loaders, tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            for idx in range(num_loaders):
                assert isinstance(self.trainer.train_dataloader.loaders[idx].loader.sampler, SequentialSampler)
            return super().training_step(batch[0], batch_idx)

        def _create_dataloader(self):
            ds = RandomDataset(32, 64)
            return DataLoader(ds, sampler=RandomSampler(ds))

        def train_dataloader(self):
            return [self._create_dataloader() for _ in range(num_loaders)]

        validation_step = None

    trainer = Trainer(default_root_dir=tmpdir, overfit_batches=1.0, max_epochs=1)
    trainer.fit(TestModel())
