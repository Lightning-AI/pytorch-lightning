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
import pickle
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict
from unittest import mock
from unittest.mock import call, Mock, PropertyMock

import pytest
import torch

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel
from tests.helpers.utils import reset_seed

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf


@mock.patch("pytorch_lightning.trainer.trainer.Trainer.node_rank", new_callable=PropertyMock)
@mock.patch("pytorch_lightning.trainer.trainer.Trainer.local_rank", new_callable=PropertyMock)
def test_can_prepare_data(local_rank, node_rank):
    dm = Mock(spec=LightningDataModule)
    dm.prepare_data_per_node = True
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    dm.prepare_data.assert_not_called()
    local_rank.return_value = 0
    assert trainer.local_rank == 0

    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()

    # local rank = 1   (False)
    dm.reset_mock()
    local_rank.return_value = 1
    assert trainer.local_rank == 1

    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    dm.reset_mock()
    dm.prepare_data_per_node = False
    node_rank.return_value = 0
    local_rank.return_value = 0

    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()

    # global rank = 1   (False)
    dm.reset_mock()
    node_rank.return_value = 1
    local_rank.return_value = 0

    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()

    node_rank.return_value = 0
    local_rank.return_value = 1

    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    dm.prepare_data_per_node = True
    local_rank.return_value = 0

    # is_overridden prepare data = True
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()


def test_hooks_no_recursion_error():
    # hooks were appended in cascade every tine a new data module was instantiated leading to a recursion error.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/3652
    class DummyDM(LightningDataModule):
        def setup(self, *args, **kwargs):
            pass

        def prepare_data(self, *args, **kwargs):
            pass

    for i in range(1005):
        dm = DummyDM()
        dm.setup()
        dm.prepare_data()


def test_helper_boringdatamodule():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup()


def test_helper_boringdatamodule_with_verbose_setup():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(["--data_dir", str(tmpdir)])
    assert args.data_dir == str(tmpdir)


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(["--data_dir", str(tmpdir)])
    dm = BoringDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    assert dm.data_dir == args.data_dir == str(tmpdir)


def test_dm_pickle_after_init():
    dm = BoringDataModule()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics["train_loss"] < 1.0


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics["train_loss"] < 1.0


def test_dm_checkpoint_save_and_load(tmpdir):
    class CustomBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log("early_stop_on", out["x"])
            return out

    class CustomBoringDataModule(BoringDataModule):
        def state_dict(self) -> Dict[str, Any]:
            return {"my": "state_dict"}

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            self.my_state_dict = state_dict

        def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            checkpoint[self.__class__.__qualname__].update({"on_save": "update"})

        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            self.checkpoint_state = checkpoint.get(self.__class__.__qualname__).copy()
            checkpoint[self.__class__.__qualname__].pop("on_save")

    reset_seed()
    dm = CustomBoringDataModule()
    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_model_summary=False,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on")],
    )

    # fit model
    with pytest.deprecated_call(
        match="`LightningDataModule.on_save_checkpoint` was deprecated in"
        " v1.6 and will be removed in v1.8. Use `state_dict` instead."
    ):
        trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__qualname__ in checkpoint
    assert checkpoint[dm.__class__.__qualname__] == {"my": "state_dict", "on_save": "update"}

    for trainer_fn in TrainerFn:
        trainer.state.fn = trainer_fn
        trainer._restore_modules_and_callbacks(checkpoint_path)
        assert dm.checkpoint_state == {"my": "state_dict", "on_save": "update"}
        assert dm.my_state_dict == {"my": "state_dict"}


def test_full_loop(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False, deterministic=True)

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(model, dm)
    assert dm.trainer is not None
    assert result[0]["val_acc"] > 0.7

    # test
    result = trainer.test(model, dm)
    assert dm.trainer is not None
    assert result[0]["test_acc"] > 0.6


@RunIf(min_gpus=1)
@mock.patch(
    "pytorch_lightning.strategies.Strategy.lightning_module",
    new_callable=PropertyMock,
)
def test_dm_apply_batch_transfer_handler(get_module_mock):
    expected_device = torch.device("cuda", 0)

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):
        rank = 0
        transfer_batch_to_device_hook_rank = None
        on_before_batch_transfer_hook_rank = None
        on_after_batch_transfer_hook_rank = None

        def on_before_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx == 0
            self.on_before_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.samples += 1
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx == 0
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            assert dataloader_idx == 0
            self.transfer_batch_to_device_hook_rank = self.rank
            self.rank += 1
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
            return batch

    dm = CurrentTestDM()
    model = BoringModel()

    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(accelerator="gpu", devices=1)
    model.trainer = trainer
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    get_module_mock.return_value = model

    trainer._data_connector.attach_datamodule(model, datamodule=dm)
    batch_gpu = trainer.strategy.batch_to_device(batch, expected_device)

    assert dm.on_before_batch_transfer_hook_rank == 0
    assert dm.transfer_batch_to_device_hook_rank == 1
    assert dm.on_after_batch_transfer_hook_rank == 2
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.ones(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


def test_dm_reload_dataloaders_every_n_epochs(tmpdir):
    """Test datamodule, where trainer argument reload_dataloaders_every_n_epochs is set to a non negative
    integer."""

    class CustomBoringDataModule(BoringDataModule):
        def __init__(self):
            super().__init__()
            self._epochs_called_for = []

        def train_dataloader(self):
            assert self.trainer.current_epoch not in self._epochs_called_for
            self._epochs_called_for.append(self.trainer.current_epoch)
            return super().train_dataloader()

    dm = CustomBoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, limit_train_batches=2, reload_dataloaders_every_n_epochs=2)
    trainer.fit(model, dm)


class DummyDS(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 100


class DummyIDS(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield 1


@pytest.mark.parametrize("iterable", (False, True))
def test_dm_init_from_datasets_dataloaders(iterable):
    ds = DummyIDS if iterable else DummyDS

    train_ds = ds()
    dm = LightningDataModule.from_datasets(train_ds, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_called_once_with(train_ds, batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)
    with pytest.raises(NotImplementedError):
        _ = dm.val_dataloader()
    with pytest.raises(NotImplementedError):
        _ = dm.test_dataloader()

    train_ds_sequence = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds_sequence, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_has_calls(
            [
                call(train_ds_sequence[0], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True),
                call(train_ds_sequence[1], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True),
            ]
        )
    with pytest.raises(NotImplementedError):
        _ = dm.val_dataloader()
    with pytest.raises(NotImplementedError):
        _ = dm.test_dataloader()

    valid_ds = ds()
    test_ds = ds()
    dm = LightningDataModule.from_datasets(val_dataset=valid_ds, test_dataset=test_ds, batch_size=2, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dl_mock.assert_called_with(valid_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        dm.test_dataloader()
        dl_mock.assert_called_with(test_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    with pytest.raises(NotImplementedError):
        _ = dm.train_dataloader()

    valid_dss = [ds(), ds()]
    test_dss = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds, valid_dss, test_dss, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dm.test_dataloader()
        dl_mock.assert_has_calls(
            [
                call(valid_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(valid_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(test_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(test_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
            ]
        )


# all args
class DataModuleWithHparams_0(LightningDataModule):
    def __init__(self, arg0, arg1, kwarg0=None):
        super().__init__()
        self.save_hyperparameters()


# single arg
class DataModuleWithHparams_1(LightningDataModule):
    def __init__(self, arg0, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(arg0)


def test_hyperparameters_saving():
    data = DataModuleWithHparams_0(10, "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"arg0": 10, "arg1": "foo", "kwarg0": "bar"})

    data = DataModuleWithHparams_1(Namespace(**{"hello": "world"}), "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"hello": "world"})

    data = DataModuleWithHparams_1({"hello": "world"}, "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"hello": "world"})

    if _OMEGACONF_AVAILABLE:
        data = DataModuleWithHparams_1(OmegaConf.create({"hello": "world"}), "foo", kwarg0="bar")
        assert data.hparams == OmegaConf.create({"hello": "world"})


def test_define_as_dataclass():
    class BoringDataModule(LightningDataModule):
        def __init__(self, foo=None):
            super().__init__()

    # makes sure that no functionality is broken and the user can still manually make
    # super().__init__ call with parameters
    # also tests all the dataclass features that can be enabled without breaking anything
    @dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
    class BoringDataModule1(BoringDataModule):
        batch_size: int
        foo: int = 2

        def __post_init__(self):
            super().__init__(foo=self.foo)

    # asserts for the different dunder methods added by dataclass, when __init__ is implemented, i.e.
    # __repr__, __eq__, __lt__, __le__, etc.
    assert BoringDataModule1(batch_size=64).foo == 2
    assert BoringDataModule1(batch_size=32)
    assert hasattr(BoringDataModule1, "__repr__")
    assert BoringDataModule1(batch_size=32) == BoringDataModule1(batch_size=32)

    # asserts inherent calling of super().__init__ in case user doesn't make the call
    @dataclass
    class BoringDataModule2(LightningDataModule):
        batch_size: int

    # asserts for the different dunder methods added by dataclass, when super class is inherently initialized, i.e.
    # __init__, __repr__, __eq__, __lt__, __le__, etc.
    assert BoringDataModule2(batch_size=32)
    assert hasattr(BoringDataModule2, "__repr__")
    assert BoringDataModule2(batch_size=32).prepare_data() is None
    assert BoringDataModule2(batch_size=32) == BoringDataModule2(batch_size=32)


def test_inconsistent_prepare_data_per_node(tmpdir):
    with pytest.raises(MisconfigurationException, match="Inconsistent settings found for `prepare_data_per_node`."):
        model = BoringModel()
        dm = BoringDataModule()
        with pytest.deprecated_call(match="prepare_data_per_node` with the trainer flag is deprecated"):
            trainer = Trainer(prepare_data_per_node=False)
        trainer.model = model
        trainer.datamodule = dm
        trainer._data_connector.prepare_data()
