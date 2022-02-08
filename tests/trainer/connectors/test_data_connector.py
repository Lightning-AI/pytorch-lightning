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
from unittest.mock import Mock

import pytest
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.connectors.data_connector import _DataHookSelector, _DataLoaderSource, warning_cache
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from tests.deprecated_api import no_warning_call
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.boring_model import RandomDataset


class NoDataLoaderModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None


@pytest.mark.parametrize(
    "instance,available",
    [
        (None, True),
        (BoringModel().train_dataloader(), True),
        (BoringModel(), True),
        (NoDataLoaderModel(), False),
        (BoringDataModule(), True),
    ],
)
def test_dataloader_source_available(instance, available):
    """Test the availability check for _DataLoaderSource."""
    source = _DataLoaderSource(instance=instance, name="train_dataloader")
    assert source.is_defined() is available


def test_dataloader_source_direct_access():
    """Test requesting a dataloader when the source is already a dataloader."""
    dataloader = BoringModel().train_dataloader()
    source = _DataLoaderSource(instance=dataloader, name="any")
    assert not source.is_module()
    assert source.is_defined()
    assert source.dataloader() is dataloader


def test_dataloader_source_request_from_module():
    """Test requesting a dataloader from a module works."""
    module = BoringModel()
    module.trainer = Trainer()
    module.foo = Mock(return_value=module.train_dataloader())

    source = _DataLoaderSource(module, "foo")
    assert source.is_module()
    module.foo.assert_not_called()
    assert isinstance(source.dataloader(), DataLoader)
    module.foo.assert_called_once()


@pytest.mark.parametrize(
    "hook_name", ("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
)
def test_datahook_selector(hook_name):
    def overridden_func(batch, *args, **kwargs):
        return batch

    def _reset_instances():
        return BoringDataModule(), BoringModel(), Trainer()

    def _no_datamodule_no_overridden():
        model, _, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=None)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        assert hook == getattr(model, hook_name)

    def _with_datamodule_no_overridden():
        model, dm, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        assert hook == getattr(model, hook_name)

    def _override_model_hook():
        model, dm, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        assert hook == getattr(model, hook_name)

    def _override_datamodule_hook():
        model, dm, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(dm, hook_name, overridden_func)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        assert hook == getattr(dm, hook_name)

    def _override_both_model_and_datamodule():
        model, dm, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, overridden_func)
        setattr(dm, hook_name, overridden_func)
        with pytest.warns(UserWarning, match=f"have overridden `{hook_name}` in both"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        warning_cache.clear()
        assert hook == getattr(dm, hook_name)

    def _with_datamodule_override_model():
        model, dm, trainer = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, overridden_func)
        with pytest.warns(UserWarning, match=f"have overridden `{hook_name}` in `LightningModule`"):
            hook = trainer._data_connector._datahook_selector.get_hook(hook_name)

        warning_cache.clear()
        assert hook == getattr(model, hook_name)

    _no_datamodule_no_overridden()
    _with_datamodule_no_overridden()
    _override_model_hook()
    _override_datamodule_hook()
    _override_both_model_and_datamodule()
    _with_datamodule_override_model()


def test_invalid_hook_passed_in_datahook_source():
    dh_selector = _DataHookSelector(BoringModel(), None)
    with pytest.raises(ValueError, match="is not a shared hook"):
        dh_selector.get_hook("setup")


def test_eval_distributed_sampler_warning(tmpdir):
    """Test that a warning is raised when `DistributedSampler` is used with evaluation."""

    model = BoringModel()
    trainer = Trainer(strategy="ddp", devices=2, accelerator="cpu", fast_dev_run=True)
    trainer._data_connector.attach_data(model)

    trainer.state.fn = TrainerFn.VALIDATING
    with pytest.warns(PossibleUserWarning, match="multi-device settings use `DistributedSampler`"):
        trainer.reset_val_dataloader(model)

    trainer.state.fn = TrainerFn.TESTING
    with pytest.warns(PossibleUserWarning, match="multi-device settings use `DistributedSampler`"):
        trainer.reset_test_dataloader(model)


@pytest.mark.parametrize("shuffle", [True, False])
def test_eval_shuffle_with_distributed_sampler_replacement(shuffle):
    """Test that shuffle is not changed if set to True."""

    class CustomModel(BoringModel):
        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), shuffle=shuffle)

    trainer = Trainer(accelerator="cpu", devices=2, strategy="ddp")
    model = CustomModel()
    trainer._data_connector.attach_data(model)
    trainer.reset_val_dataloader(model)
    assert trainer.val_dataloaders[0].sampler.shuffle == shuffle
