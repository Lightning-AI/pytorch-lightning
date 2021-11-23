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
from unittest import mock
from unittest.mock import Mock, PropertyMock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.connectors.data_connector import _DataLoaderSource, warning_cache
from tests.deprecated_api import no_warning_call
from tests.helpers import BoringDataModule, BoringModel


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
@mock.patch("pytorch_lightning.accelerators.accelerator.Accelerator.lightning_module", new_callable=PropertyMock)
def test_shared_datahook_call(module_mock, hook_name):
    class CustomBatch:
        def __init__(self):
            self.x = torch.randn(3, 4)
            self.instance_type = []

        def to(self, device):
            self.x = self.x.to(device)

    def overridden_model_func(batch, *args, **kwargs):
        batch.instance_type.append("model")
        return batch

    def overridden_dm_func(batch, *args, **kwargs):
        batch.instance_type.append("datamodule")
        return batch

    def _reset_instances():
        dm = BoringDataModule()
        model = BoringModel()
        trainer = Trainer()
        model.trainer = trainer
        module_mock.return_value = model
        batch = CustomBatch()
        return model, dm, trainer, batch

    def _no_datamodule_no_overridden():
        model, _, trainer, batch = _reset_instances()
        setattr(model, hook_name, Mock(getattr(model, hook_name)))
        trainer._data_connector.attach_datamodule(model, datamodule=None)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            trainer.accelerator.batch_to_device(batch, torch.device("cpu"))

        assert getattr(model, hook_name).call_count == 1

    def _with_datamodule_no_overridden():
        model, dm, trainer, batch = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, Mock(getattr(model, hook_name)))
        setattr(dm, hook_name, Mock(getattr(dm, hook_name)))
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            trainer.accelerator.batch_to_device(batch, torch.device("cpu"))

        assert getattr(model, hook_name).call_count == 1
        assert getattr(dm, hook_name).call_count == 0

    def _override_model_hook():
        model, dm, trainer, batch = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, overridden_model_func)
        setattr(dm, hook_name, Mock(getattr(dm, hook_name)))
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            trainer.accelerator.batch_to_device(batch, torch.device("cpu"))

        assert batch.instance_type == ["model"]
        assert getattr(dm, hook_name).call_count == 0

    def _override_datamodule_hook():
        model, dm, trainer, batch = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, Mock(getattr(model, hook_name)))
        setattr(dm, hook_name, overridden_dm_func)
        with no_warning_call(match="have overridden `{hook_name}` in both"):
            trainer.accelerator.batch_to_device(batch, torch.device("cpu"))

        assert batch.instance_type == ["datamodule"]
        assert getattr(model, hook_name).call_count == 0

    def _override_both_model_and_datamodule():
        model, dm, trainer, batch = _reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, overridden_model_func)
        setattr(dm, hook_name, overridden_dm_func)
        with pytest.warns(UserWarning, match=f"have overridden `{hook_name}` in both"):
            trainer.accelerator.batch_to_device(batch, torch.device("cpu"))

        warning_cache.clear()

        assert batch.instance_type == ["datamodule"]

    _no_datamodule_no_overridden()
    _with_datamodule_no_overridden()
    _override_model_hook()
    _override_datamodule_hook()
    _override_both_model_and_datamodule()
