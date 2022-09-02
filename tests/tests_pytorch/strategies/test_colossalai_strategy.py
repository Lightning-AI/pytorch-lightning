import contextlib
import json
import logging
import os
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from pytorch_lightning.strategies import ColossalAIStrategy
from pytorch_lightning.strategies.colossalai import _COLOSSALAI_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf

if _COLOSSALAI_AVAILABLE:
    from colossalai.nn.optimizer import HybridAdam
    from colossalai.zero import ZeroOptimizer


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()

    def configure_optimizers(self):
        optimizer = HybridAdam(self.layer.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class ModelParallelBoringModelNoSchedulers(ModelParallelBoringModel):
    def configure_optimizers(self):
        return HybridAdam(self.layer.parameters(), lr=1e-3)


@RunIf(min_cuda_gpus=1, standalone=True, colossalai=True)
def test_colossalai_optimizer(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        strategy=ColossalAIStrategy(),
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(AssertionError, match='ColossalAIStrategy only supports colossalai.nn.optimizer.CPUAdam and colossalai.nn.optimizer.HybridAdam'):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, standalone=True, colossalai=True)
def test_warn_colossalai_ignored(tmpdir):
    class TestModel(ModelParallelBoringModel):
        def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
            return loss.backward()

    model = TestModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        strategy=ColossalAIStrategy(),
        devices=1,
        track_grad_norm=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    from pytorch_lightning.plugins.precision.colossalai import warning_cache

    with pytest.warns(UserWarning, match="will be ignored since ColossalAI handles the backward"):
        trainer.fit(model)
    print(warning_cache)
    assert any("track_grad_norm=2.0)' but this is not supported" in w for w in warning_cache)


def _assert_save_model_is_equal(model, tmpdir, trainer):
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    trainer.strategy.barrier()

    # carry out the check only on rank 0
    if trainer.is_global_zero:
        state_dict = torch.load(checkpoint_path)

        # Assert model parameters are identical after loading
        for orig_param, saved_model_param in zip(model.parameters(), state_dict.values()):
            saved_model_param = saved_model_param.to(dtype=orig_param.dtype, device=orig_param.device)
            assert torch.equal(orig_param, saved_model_param)


# @RunIf(min_cuda_gpus=1, standalone=True, colossalai=True)
# def test_colossalai_run_configure_optimizers(tmpdir):
#     """Test end to end that deepspeed works with defaults (without ZeRO as that requires compilation), whilst using
#     configure_optimizers for optimizers and schedulers."""

#     class TestCB(Callback):
#         def on_train_start(self, trainer, pl_module) -> None:
#             assert isinstance(trainer.optimizers[0], ZeroOptimizer)
#             assert isinstance(trainer.optimizers[0].optim, HybridAdam)
#             assert isinstance(trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.StepLR)
#             # check that the lr_scheduler config was preserved
#             assert trainer.lr_scheduler_configs[0].name == "Sean"

#     class TestModel(ModelParallelBoringModel):
#         def configure_optimizers(self):
#             [optimizer], [scheduler] = super().configure_optimizers()
#             return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "name": "Sean"}}

#     model = TestModel()
#     lr_monitor = LearningRateMonitor()
#     trainer = Trainer(
#         strategy=ColossalAIStrategy(),  # disable ZeRO so our optimizers are not wrapped
#         default_root_dir=tmpdir,
#         devices=1,
#         fast_dev_run=True,
#         callbacks=[TestCB(), lr_monitor],
#         enable_progress_bar=False,
#         enable_model_summary=False,
#     )
#     trainer.fit(model)

#     assert lr_monitor.lrs == {"Sean": [1e-3]}

#     _assert_save_model_is_equal(model, tmpdir, trainer)
