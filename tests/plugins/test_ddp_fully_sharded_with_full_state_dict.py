import os
from typing import Any, Dict
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
    from fairscale.nn import auto_wrap, default_auto_wrap_policy, FullyShardedDataParallel, wrap


@RunIf(fairscale_fully_sharded=True)
def test_sharded_ddp_choice(tmpdir):
    """
    Test to ensure that plugin is correctly chosen
    """
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        plugins="fsdp_full_state_dict",
    )
    assert isinstance(trainer.accelerator.training_type_plugin, DDPFullyShardedPlugin)


@RunIf(amp_apex=True, fairscale_fully_sharded=True)
def test_invalid_apex_sharded(tmpdir):
    """
    Test to ensure that we raise an error when we try to use apex and sharded
    """

    model = BoringModel()
    with pytest.raises(
        MisconfigurationException,
        match="Sharded Plugins are not supported with Apex AMP",
    ):
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            plugins="fsdp_full_state_dict",
            precision=16,
            amp_backend="apex",
        )

        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
@RunIf(amp_native=True, fairscale_fully_sharded=True)
def test_ddp_choice_sharded_amp(device_count_mock, mock_cuda_available, tmpdir):
    """
    Test to ensure that plugin native amp plugin is correctly chosen when using sharded
    """
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        gpus=1,
        precision=16,
        plugins="fsdp_full_state_dict",
    )

    assert isinstance(trainer.accelerator.training_type_plugin, DDPFullyShardedPlugin)
    assert isinstance(trainer.accelerator.precision_plugin, FullyShardedNativeMixedPrecisionPlugin)


@RunIf(min_gpus=1, skip_windows=True, fairscale_fully_sharded=True)
def test_fully_sharded_plugin_checkpoint(tmpdir):
    """
    Test to ensure that checkpoint is saved correctly when using a single GPU.
    """

    class TestModel(BoringModel):

        def configure_sharded_model(self) -> None:
            self.layer = wrap(
                torch.nn.Sequential(
                    wrap(torch.nn.Linear(32, 32)),
                    torch.nn.ReLU(),
                    wrap(torch.nn.Linear(32, 2)),
                )
            )

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

        def on_train_start(self) -> None:
            assert isinstance(self.layer, FullyShardedDataParallel)
            assert isinstance(self.layer.module[0], FullyShardedDataParallel)
            assert isinstance(self.layer.module[2], FullyShardedDataParallel)
            # root should not be resharding
            assert self.layer.reshard_after_forward is False
            # Assert that the nested layers are set reshard_after_forward to True
            assert self.layer.module[0].reshard_after_forward is True
            assert self.layer.module[2].reshard_after_forward is True

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        plugins="fsdp_full_state_dict",
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)

    _assert_save_equality(tmpdir, trainer, cls=TestModel)


@RunIf(min_gpus=2, skip_windows=True, fairscale_fully_sharded=True, special=False)
def test_fully_sharded_plugin_multi_gpu(tmpdir):
    """
    Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run.
    """

    class TestModel(BoringModel):

        def configure_sharded_model(self) -> None:
            self.layer = wrap(self.layer)

        def configure_optimizers(self):
            return torch.optim.SGD(self.trainer.model.parameters(), lr=0.1)

    ck = ModelCheckpoint(save_last=True)
    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        plugins="fsdp_full_state_dict",
        max_epochs=5,
        precision=16,
        callbacks=ck,
    )

    trainer.fit(model)
    trainer.test(model)
    trainer.test(ckpt_path=ck.last_model_path)
    trainer.validate()
    trainer.validate(ckpt_path=ck.last_model_path)
    trainer.predict(dataloaders=model.val_dataloader())

    _assert_save_equality(tmpdir, trainer, cls=TestModel)


def _assert_save_equality(tmpdir, trainer, cls=BoringModel):
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.lightning_module.state_dict()

    if trainer.global_rank == 0:
        saved_model = cls.load_from_checkpoint(checkpoint_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)
