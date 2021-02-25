from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.nn import DataParallel

from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.base import warning_cache
from pytorch_lightning.overrides.data_parallel import (
    LightningParallelModule,
    python_scalar_to_tensor,
    unsqueeze_scalar_tensor,
)
from pytorch_lightning.trainer.states import RunningStage
from tests.helpers import BoringModel


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
@pytest.mark.parametrize("stage", [
    ("training", "training_step"),
    ("testing", "test_step"),
    ("validating", "validation_step"),
    ("predicting", "predict"),
])
def test_lightning_wrapper_module_methods(wrapper_class, stage):
    """ Test that the LightningWrapper redirects .forward() to the LightningModule methods. """
    pl_module = MagicMock()
    wrapped_module = wrapper_class(pl_module)

    batch = torch.rand(5)
    batch_idx = 3

    prop, step = stage
    pl_module.trainer.sanity_checking = False
    for p in ("training", "testing", "validating", "predicting"):
        setattr(pl_module.trainer, p, p == prop)

    wrapped_module(batch, batch_idx)

    getattr(pl_module, step).assert_called_with(batch, batch_idx)


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
@pytest.mark.parametrize("stage", [
    ("training", "training_step"),
    ("testing", "test_step"),
    ("validating", "validation_step"),
])
def test_lightning_wrapper_module_warn_none_output(wrapper_class, stage):
    """ Test that the LightningWrapper module warns about forgotten return statement. """
    warning_cache.clear()
    pl_module = MagicMock()

    prop, step = stage
    pl_module.trainer.sanity_checking = False
    for p in ("training", "testing", "validating", "predicting"):
        setattr(pl_module.trainer, p, p == prop)

    wrapped_module = wrapper_class(pl_module)

    getattr(pl_module, step).return_value = None

    with pytest.warns(UserWarning, match=f"Your {step} returned None"):
        wrapped_module()


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
def test_lightning_wrapper_module_no_warn(wrapper_class):
    warning_cache.clear()
    pl_module = MagicMock()

    pl_module.trainer.sanity_checking = False
    pl_module.trainer.training = False
    pl_module.trainer.testing = False
    pl_module.trainer.validating = False
    pl_module.trainer.predicting = False

    wrapped_module = wrapper_class(pl_module)

    with pytest.warns(None) as record:
        wrapped_module()
        pl_module.assert_called()
        assert not record


@pytest.mark.parametrize(
    "inp,expected", [
        [torch.tensor(1.0), torch.tensor([1.0])],
        [torch.tensor([2.0]), torch.tensor([2.0])],
        [torch.ones(3, 4, 5), torch.ones(3, 4, 5)],
    ]
)
def test_unsqueeze_scalar_tensor(inp, expected):
    """ Test that the utility function unsqueezes only scalar tensors. """
    assert torch.all(unsqueeze_scalar_tensor(inp).eq(expected))


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
    model.trainer = Mock()
    model.trainer._running_stage = RunningStage.TRAINING
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


@pytest.mark.parametrize(
    "inp,expected", [
        [1.0, torch.tensor([1.0])],
        [2, torch.tensor([2.0])],
        [True, torch.tensor([True])],
    ]
)
def test_python_scalar_to_tensor(inp, expected):
    assert torch.all(python_scalar_to_tensor(inp).eq(expected))


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda", 0)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_lightning_parallel_module_python_scalar_conversion(device):
    """ Test that LightningParallelModule can convert Python scalars to tensors. """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            # PyTorch DP does not support Python scalars, Lightning converts them to tensors
            output.update({"python scalar": 12.3})
            return output

    model = TestModel()
    model.to(device)
    model.trainer = Mock()
    model.trainer._running_stage = RunningStage.TRAINING
    batch = torch.rand(2, 32).to(device)
    batch_idx = 0

    wrapped_model = LightningParallelModule(model)
    output = wrapped_model(batch, batch_idx)
    assert output["python scalar"] == torch.tensor([12.3], device=device)
