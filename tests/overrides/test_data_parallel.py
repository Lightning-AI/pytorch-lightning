from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning.overrides.data_parallel import LightningDistributedModule
from pytorch_lightning.trainer.states import RunningStage


def test_lightning_distributed_module_methods():
    """ Test that the LightningDistributedModule redirects .forward() to the LightningModule methods. """
    pl_module = MagicMock()
    dist_module = LightningDistributedModule(pl_module)

    batch = torch.rand(5)
    batch_idx = 3

    pl_module.running_stage = RunningStage.TRAINING
    dist_module(batch, batch_idx)
    pl_module.training_step.assert_called_with(batch, batch_idx)

    pl_module.running_stage = RunningStage.TESTING
    dist_module(batch, batch_idx)
    pl_module.test_step.assert_called_with(batch, batch_idx)

    pl_module.running_stage = RunningStage.EVALUATING
    dist_module(batch, batch_idx)
    pl_module.validation_step.assert_called_with(batch, batch_idx)


def test_lightning_distributed_module_warn_none_output():
    """ Test that the LightningDistributedModule warns about forgotten return statement. """
    pl_module = MagicMock()
    dist_module = LightningDistributedModule(pl_module)

    pl_module.training_step.return_value = None
    pl_module.validation_step.return_value = None
    pl_module.test_step.return_value = None

    with pytest.warns(UserWarning, match="Your training_step returned None"):
        pl_module.running_stage = RunningStage.TRAINING
        dist_module()

    with pytest.warns(UserWarning, match="Your test_step returned None"):
        pl_module.running_stage = RunningStage.TESTING
        dist_module()

    with pytest.warns(UserWarning, match="Your validation_step returned None"):
        pl_module.running_stage = RunningStage.EVALUATING
        dist_module()
