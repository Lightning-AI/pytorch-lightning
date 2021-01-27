from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning.overrides.data_parallel import LightningDistributedModule, LightningParallelModule


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
