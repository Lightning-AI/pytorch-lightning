import os
from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins.precision import ColossalAIPrecisionPlugin
from pytorch_lightning.strategies import ColossalAIStrategy
from pytorch_lightning.strategies.colossalai import _COLOSSALAI_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf

if _COLOSSALAI_AVAILABLE:
    from colossalai.nn.optimizer import HybridAdam


def test_invalid_colosalai(monkeypatch):
    import pytorch_lightning.strategies.colossalai as colossal_strategy

    monkeypatch.setattr(colossal_strategy, "_COLOSSALAI_AVAILABLE", False)
    with pytest.raises(MisconfigurationException):
        ColossalAIStrategy()


@RunIf(colossalai=True)
def test_colossalai_strategy_with_trainer(tmpdir):
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, strategy=ColossalAIStrategy())

    assert isinstance(trainer.strategy, ColossalAIStrategy)
    assert isinstance(trainer.strategy.precision_plugin, ColossalAIPrecisionPlugin)


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
    with pytest.raises(
        AssertionError,
        match="ColossalAIStrategy only supports colossalai.nn.optimizer.CPUAdam and colossalai.nn.optimizer.HybridAdam",
    ):
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


class ModelParallelClassificationModel(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr
        self.layers = None

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def build_layers(self) -> nn.Module:
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(32, 32))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 3))
        return nn.Sequential(*layers)

    def configure_sharded_model(self) -> None:
        if self.layers is None:
            self.layers = self.build_layers()

    def forward(self, x):
        x = self.layers(x)
        logits = F.softmax(x, dim=1)
        return logits

    def configure_optimizers(self):
        optimizer = HybridAdam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)


@RunIf(min_cuda_gpus=2, standalone=True, colossalai=True)
def test_multi_gpu_model_colossalai_fit_only(tmpdir):
    dm = ClassifDataModule()
    model = ModelParallelClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, devices=2, strategy=ColossalAIStrategy())
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, colossalai=True)
def test_multi_gpu_model_colossalai_test_only(tmpdir):
    dm = ClassifDataModule()
    model = ModelParallelClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, devices=2, strategy=ColossalAIStrategy())
    trainer.test(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, colossalai=True)
def test_multi_gpu_model_colossalai_fit_test(tmpdir):
    seed_everything(4321)
    dm = ClassifDataModule()
    model = ModelParallelClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, devices=2, strategy=ColossalAIStrategy(initial_scale=32))
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)

    for out in result:
        assert out["test_acc"] > 0.7
