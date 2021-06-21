import os
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPFullyShardedPlugin, FullyShardedNativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _FAIRSCALE_FULLY_SHARDED_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf

if _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
    from fairscale.nn import FullyShardedDataParallel, wrap


def test_invalid_on_cpu(tmpdir):
    """
    Test to ensure that to raise Misconfiguration for FSDP on CPU.
    """
    with pytest.raises(
        MisconfigurationException,
        match="You selected accelerator to be `ddp_fully_sharded`, but GPU is not available.",
    ):
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            plugins="fsdp",
        )
        assert isinstance(trainer.accelerator.training_type_plugin, DDPFullyShardedPlugin)
        trainer.accelerator.setup_environment()


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@RunIf(amp_apex=True, fairscale_fully_sharded=True)
def test_invalid_apex_sharded(device_count_mock, mock_cuda_available, tmpdir):
    """
    Test to ensure that we raise an error when we try to use apex and fully sharded
    """
    with pytest.raises(
        MisconfigurationException,
        match="Sharded Plugin is not supported with Apex AMP",
    ):
        Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            plugins="fsdp",
            gpus=1,
            precision=16,
            amp_backend="apex",
        )


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@RunIf(amp_native=True, fairscale_fully_sharded=True)
def test_fsdp_with_sharded_amp(device_count_mock, mock_cuda_available, tmpdir):
    """
    Test to ensure that plugin native amp plugin is correctly chosen when using sharded
    """
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        plugins="fsdp",
        gpus=1,
        precision=16,
    )
    assert isinstance(trainer.accelerator.training_type_plugin, DDPFullyShardedPlugin)
    assert isinstance(trainer.accelerator.precision_plugin, FullyShardedNativeMixedPrecisionPlugin)


class TestFSDPModel(BoringModel):

    def setup(self, stage: str) -> None:
        if stage != "fit":
            # when running stages like test, validate, and predict, we will skip setting up,
            # will directly use the module itself unless we load from checkpoint
            return
        # resetting call_configure_sharded_model_hook attribute so that we could call
        # configure sharded model
        self.call_configure_sharded_model_hook = False
        # for loading full state dict, we first need to create a new unwrapped model
        # to load state dict and then wrapping
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def configure_sharded_model(self) -> None:
        for i, layer in enumerate(self.layer):
            if i % 2 == 0:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # when loading full state dict, we first need to create a new unwrapped model
        self.setup("fit")

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


@RunIf(
    min_gpus=1,
    skip_windows=True,
    fairscale_fully_sharded=True,
    amp_native=True,
    special=True,
)
def test_fully_sharded_plugin_checkpoint(tmpdir):
    """
    Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run.
    """

    model = TestFSDPModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        plugins="fsdp",
        precision=16,
        max_epochs=1,
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


@RunIf(
    min_gpus=2,
    skip_windows=True,
    fairscale_fully_sharded=True,
    amp_native=True,
    special=True,
)
def test_fully_sharded_plugin_checkpoint_multi_gpus(tmpdir):
    """
    Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run.
    """

    model = TestFSDPModel()
    ck = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=2,
        plugins="fsdp",
        precision=16,
        max_epochs=1,
        callbacks=[ck],
    )
    _run_multiple_stages(trainer, model)


def _assert_save_equality(trainer, ckpt_path, cls=TestFSDPModel):
    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.accelerator.lightning_module_state_dict()

    if trainer.is_global_zero:
        saved_model = cls.load_from_checkpoint(ckpt_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)

    model_call_configure_sharded_model_hook = getattr(model, "call_configure_sharded_model_hook", False)
    trainer_accelerator_call_configure_sharded_model_hook = (trainer.accelerator.call_configure_sharded_model_hook)

    model_path = (model_path if model_path else trainer.checkpoint_callback.last_model_path)

    assert model_call_configure_sharded_model_hook
    assert not trainer_accelerator_call_configure_sharded_model_hook
    trainer.save_checkpoint(model_path, weights_only=True)

    _assert_save_equality(trainer, model_path, cls=TestFSDPModel)

    # Test entry point
    trainer.test(model)  # model is wrapped, will not call configure_shared_model

    # provide model path, will create a new unwrapped model and load and then call configure_shared_model to wrap
    trainer.test(ckpt_path=model_path)
