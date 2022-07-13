import os
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from tests_pytorch.helpers.runif import RunIf

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap


@RunIf(min_torch="1.12dev")
def test_invalid_on_cpu(tmpdir):
    """Test to ensure that to raise Misconfiguration for Native FSDP on CPU."""
    with pytest.raises(
        MisconfigurationException,
        match=f"You selected strategy to be `{DDPFullyShardedNativeStrategy.strategy_name}`, "
        "but GPU accelerator is not used.",
    ):
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy="fsdp_native")
        assert isinstance(trainer.strategy, DDPFullyShardedNativeStrategy)
        trainer.strategy.setup_environment()


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@RunIf(min_torch="1.12dev")
def test_fsdp_with_sharded_amp(device_count_mock, mock_cuda_available, tmpdir):
    """Test to ensure that plugin native amp plugin raises Misconfiguration error."""
    with pytest.raises(
        MisconfigurationException, match="DDPFullyShardedNativeStrategy currently doesn't support Mixed Precision"
    ):
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            strategy="fsdp_native",
            accelerator="gpu",
            devices=1,
            precision=16,
        )
        assert isinstance(trainer.strategy, DDPFullyShardedNativeStrategy)


class TestFSDPModel(BoringModel):
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

    def on_train_start(self) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_start(self) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_start(self) -> None:
        self._assert_layer_fsdp_instance()

    def on_prediction_start(self) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, FullyShardedDataParallel)
        assert isinstance(self.layer.module[0], FullyShardedDataParallel)
        assert isinstance(self.layer.module[2], FullyShardedDataParallel)
        # root should not be resharding
        assert self.layer.reshard_after_forward is False
        # Assert that the nested layers are set reshard_after_forward to True
        assert self.layer.module[0].reshard_after_forward is True
        assert self.layer.module[2].reshard_after_forward is True


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12dev")
def test_fully_sharded_native_strategy_sync_batchnorm(tmpdir):
    """Test to ensure that sync_batchnorm works when using fsdp_native and GPU, and all stages can be run."""

    model = TestFSDPModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy="fsdp_native",
        precision=16,
        max_epochs=1,
        sync_batchnorm=True,
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12dev")
def test_fully_sharded_native_strategy_checkpoint(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""

    model = TestFSDPModel()
    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="gpu", devices=1, strategy="fsdp_native", precision=16, max_epochs=1
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12dev")
def test_fully_sharded_native_strategy_checkpoint_multi_gpus(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""

    model = TestFSDPModel()
    ck = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy="fsdp_native",
        precision=16,
        max_epochs=1,
        callbacks=[ck],
    )
    _run_multiple_stages(trainer, model)


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)

    model_path = model_path if model_path else trainer.checkpoint_callback.last_model_path

    trainer.save_checkpoint(model_path, weights_only=True)

    _assert_save_equality(trainer, model_path, cls=TestFSDPModel)

    # Test entry point
    trainer.test(model)  # model is wrapped, will not call configure_shared_model

    # provide model path, will create a new unwrapped model and load and then call configure_shared_model to wrap
    trainer.test(ckpt_path=model_path)


def _assert_save_equality(trainer, ckpt_path, cls=TestFSDPModel):
    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.is_global_zero:
        saved_model = cls.load_from_checkpoint(ckpt_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)
