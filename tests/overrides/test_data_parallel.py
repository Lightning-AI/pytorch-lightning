from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import DataParallel

from pytorch_lightning.overrides.data_parallel import LightningDistributedModule, LightningParallelModule, warning_cache
from tests.base import BoringModel


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
def test_lightning_wrapper_module_methods(wrapper_class):
    """ Test that the LightningWrapper redirects .forward() to the LightningModule methods. """
    pl_module = MagicMock()
    wrapped_module = wrapper_class(pl_module)

    batch = torch.rand(5)
    batch_idx = 3

    pl_module.training = True
    pl_module.testing = False
    wrapped_module(batch, batch_idx)
    pl_module.training_step.assert_called_with(batch, batch_idx)

    pl_module.training = False
    pl_module.testing = True
    wrapped_module(batch, batch_idx)
    pl_module.test_step.assert_called_with(batch, batch_idx)

    pl_module.training = False
    pl_module.testing = False
    wrapped_module(batch, batch_idx)
    pl_module.validation_step.assert_called_with(batch, batch_idx)


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
def test_lightning_wrapper_module_warn_none_output(wrapper_class):
    """ Test that the LightningWrapper module warns about forgotten return statement. """
    warning_cache.clear()
    pl_module = MagicMock()
    wrapped_module = wrapper_class(pl_module)

    pl_module.training_step.return_value = None
    pl_module.validation_step.return_value = None
    pl_module.test_step.return_value = None

    with pytest.warns(UserWarning, match="Your training_step returned None"):
        pl_module.training = True
        pl_module.testing = False
        wrapped_module()

    with pytest.warns(UserWarning, match="Your test_step returned None"):
        pl_module.training = False
        pl_module.testing = True
        wrapped_module()

    with pytest.warns(UserWarning, match="Your validation_step returned None"):
        pl_module.training = False
        pl_module.testing = False
        wrapped_module()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-gpu machine")
def test_lightning_parallel_module_unsqueeze_scalar():
    """ Test that LightningParallelModule takes care of un-squeezeing 0-dim tensors. """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            loss = output["loss"]
            loss = loss.squeeze()
            assert loss.dim() == 0
            # PyTorch usually warns about 0-dim tensors returned in DP
            return {"loss": loss}

    model = TestModel()
    batch = torch.rand(2, 32).cuda()
    batch_idx = 0

    wrapped_model = LightningParallelModule(model).cuda()
    dp_module = DataParallel(wrapped_model, device_ids=[0, 1])

    output = wrapped_model(batch, batch_idx)
    assert output["loss"].dim() == 1

    with pytest.warns(None) as record:
        output = dp_module(batch, batch_idx)

    assert output["loss"].dim() == 1
    assert not record


@pytest.mark.parametrize("device", [
    torch.device("cpu"),
    torch.device("cuda", 0)
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_lightning_parallel_module_python_scalar_conversion(device):
    """ Test that LightningParallelModule can convert Python scalars to tensors. """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            # PyTorch DP does not support Python scalars, Lightning converts them to tensors
            output.update({"python scalar": 12.3})
            return output

    model = TestModel().to(device)
    batch = torch.rand(2, 32).cuda()
    batch_idx = 0

    wrapped_model = LightningParallelModule(model)
    output = wrapped_model(batch, batch_idx)
    assert output["python scalar"] == torch.tensor([12.3], device=device)
