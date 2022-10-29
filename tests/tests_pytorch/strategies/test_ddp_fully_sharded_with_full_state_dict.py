import os
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch

from lightning_lite.strategies.fairscale import _FAIRSCALE_AVAILABLE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins import FullyShardedNativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPFullyShardedStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn import FullyShardedDataParallel, wrap


def test_invalid_on_cpu(tmpdir):
    """Test to ensure that to raise Misconfiguration for FSDP on CPU."""
    with pytest.raises(
        MisconfigurationException, match="You selected strategy to be `ddp_fully_sharded`, but GPU is not available."
    ):
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy="fsdp")
        assert isinstance(trainer.strategy, DDPFullyShardedStrategy)
        trainer.strategy.setup_environment()


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@RunIf(fairscale=True)
def test_fsdp_with_sharded_amp(cuda_count_1, tmpdir):
    """Test to ensure that plugin native amp plugin is correctly chosen when using sharded."""
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, strategy="fsdp", accelerator="gpu", devices=1, precision=16
    )
    assert isinstance(trainer.strategy, DDPFullyShardedStrategy)
    assert isinstance(trainer.strategy.precision_plugin, FullyShardedNativeMixedPrecisionPlugin)


class TestFSDPModelManualWrapped(BoringModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer: Optional[torch.nn.Module] = None

    def _init_model(self) -> None:
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def setup(self, stage: str) -> None:
        if self.layer is None:
            self._init_model()

    def configure_sharded_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i % 2 == 0:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # when loading full state dict, we first need to create a new unwrapped model
        self._init_model()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def on_train_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_prediction_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, FullyShardedDataParallel)
        assert isinstance(self.layer.module[0], FullyShardedDataParallel)
        assert isinstance(self.layer.module[2], FullyShardedDataParallel)

        # Assert that the nested layers are set reshard_after_forward to True
        assert self.layer.module[0].reshard_after_forward
        assert self.layer.module[2].reshard_after_forward

        if isinstance(self.trainer.precision_plugin, FullyShardedNativeMixedPrecisionPlugin):
            assert self.layer.mixed_precision
            assert self.layer.module[0].mixed_precision
            assert self.layer.module[2].mixed_precision


class TestFSDPModelAutoWrapped(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_optimizers(self):
        return torch.optim.SGD(self.trainer.model.parameters(), lr=0.1)

    def on_train_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def on_prediction_batch_end(self, *_, **__) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.trainer.model, FullyShardedDataParallel)
        # `disable_reshard_on_root=True` (default) in FSDP which turns-off resharding
        assert not self.trainer.model.reshard_after_forward

        if isinstance(self.trainer.precision_plugin, FullyShardedNativeMixedPrecisionPlugin):
            assert self.trainer.model.mixed_precision


@RunIf(min_cuda_gpus=1, standalone=True, fairscale=True)
def test_fully_sharded_strategy_checkpoint(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""

    model = TestFSDPModelManualWrapped()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=1,
        strategy="fsdp",
        precision=16,
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


@RunIf(min_cuda_gpus=2, standalone=True, fairscale=True)
@pytest.mark.parametrize(
    "model, strategy",
    [
        (TestFSDPModelManualWrapped(), DDPFullyShardedStrategy(min_num_params=2)),
        (TestFSDPModelAutoWrapped(), "fsdp"),
    ],
)
def test_fully_sharded_strategy_checkpoint_multi_gpus(tmpdir, model, strategy):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""

    ck = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision=16,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        callbacks=[ck],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    _run_multiple_stages(trainer, model)


def _assert_save_equality(trainer, ckpt_path, cls=TestFSDPModelManualWrapped):
    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.is_global_zero:
        saved_model = cls.load_from_checkpoint(ckpt_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)

    model_path = model_path if model_path else trainer.checkpoint_callback.last_model_path

    trainer.save_checkpoint(model_path, weights_only=True)

    _assert_save_equality(trainer, model_path, cls=model.__class__)

    # Test entry point
    if model.__class__ is TestFSDPModelAutoWrapped:
        model = TestFSDPModelAutoWrapped()
    trainer.test(model)  # model is wrapped, will not call configure_shared_model

    # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
    if model.__class__ is TestFSDPModelAutoWrapped:
        model = TestFSDPModelAutoWrapped()
    trainer.test(model, ckpt_path=model_path)

    # Predict entry point
    if model.__class__ is TestFSDPModelAutoWrapped:
        model = TestFSDPModelAutoWrapped()

    if model.__class__ is TestFSDPModelAutoWrapped:
        model = TestFSDPModelAutoWrapped()
    trainer.predict(model)  # model is wrapped, will not call `configure_sharded_model`

    # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
    if model.__class__ is TestFSDPModelAutoWrapped:
        model = TestFSDPModelAutoWrapped()
    trainer.predict(model, ckpt_path=model_path)


@RunIf(min_cuda_gpus=1, standalone=True, fairscale=True)
def test_fsdp_gradient_clipping_raises(tmpdir):
    """Test to ensure that an exception is raised when clipping gradients by value with FSDP."""
    model = TestFSDPModelManualWrapped()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy="fsdp",
        fast_dev_run=True,
        accelerator="gpu",
        devices=1,
        precision=16,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(
        MisconfigurationException, match="gradient_clip_algorithm='norm'` is currently not supported for `FullySharded"
    ):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, fairscale_fully_sharded=True)
def test_fsdp_rewrap_limitation(tmpdir):
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=1,
        max_steps=1,
        limit_val_batches=0,
        limit_test_batches=1,
        strategy="fsdp",
    )
    model = TestFSDPModelAutoWrapped()
    trainer.fit(model)

    with pytest.raises(MisconfigurationException, match="Using the same instance of model .* not supported"):
        trainer.test(model)
