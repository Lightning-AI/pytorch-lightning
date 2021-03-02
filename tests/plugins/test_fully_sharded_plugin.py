import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import FullyShardedNativeMixedPrecisionPlugin, FullyShardedPlugin
from pytorch_lightning.utilities import _FAIRSCALE_FULLY_SHARDED_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf

if _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
    from fairscale.nn import auto_wrap, FullyShardedDataParallel


@pytest.mark.parametrize(["plugin"], [("ddp_fully_sharded", )])
@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
def test_sharded_ddp_choice(tmpdir, plugin):
    """
        Test to ensure that plugin is correctly chosen
    """

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            if plugin == 'ddp_fully_sharded':
                assert isinstance(trainer.accelerator.training_type_plugin, FullyShardedPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        plugins=plugin,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
@RunIf(amp_apex=True)
def test_invalid_apex_sharded(tmpdir):
    """
        Test to ensure that we raise an error when we try to use apex and sharded
    """

    model = BoringModel()
    with pytest.raises(MisconfigurationException, match='Sharded Plugins are not supported with Apex AMP'):
        trainer = Trainer(
            fast_dev_run=True,
            plugins='ddp_fully_sharded',
            precision=16,
            amp_backend='apex',
        )

        trainer.fit(model)


@pytest.mark.parametrize(["plugin"], [("ddp_fully_sharded", )])
@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch('torch.cuda.device_count', return_value=1)
@mock.patch('torch.cuda.is_available', return_value=True)
@RunIf(amp_native=True)
def test_ddp_choice_sharded_amp(device_count_mock, mock_cuda_available, plugin, tmpdir):
    """
        Test to ensure that plugin native amp plugin is correctly chosen when using sharded
    """

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            if plugin == 'ddp_fully_sharded':
                assert isinstance(trainer.accelerator.training_type_plugin, FullyShardedPlugin)
            assert isinstance(trainer.accelerator.precision_plugin, FullyShardedNativeMixedPrecisionPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=1,
        precision=16,
        plugins=plugin,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
@RunIf(min_gpus=1, skip_windows=True)
def test_fully_sharded_plugin_checkpoint(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly when using a single GPU.
    """
    model = BoringModel()
    trainer = Trainer(
        gpus=1,
        plugins='ddp_fully_sharded',
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)

    _assert_save_equality(tmpdir, trainer)


@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
@RunIf(min_gpus=1, skip_windows=True)
def test_fully_sharded_plugin_checkpoint_autowrap(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly when using auto_wrap.
    """

    class TestModel(BoringModel):

        def on_distributed_model_setup(self) -> None:
            self.layer = auto_wrap(self.layer, min_num_params=1)

        def on_train_start(self) -> None:
            assert isinstance(self.layer, FullyShardedDataParallel)
            assert isinstance(self.accelerator_model, FullyShardedDataParallel)

    model = TestModel()

    trainer = Trainer(
        gpus=1,
        plugins='ddp_fully_sharded',
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)

    _assert_save_equality(tmpdir, trainer)


@pytest.mark.skipif(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE, reason="Fairscale is not available")
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
@RunIf(min_gpus=2, skip_windows=True)
def test_fully_sharded_plugin_checkpoint_multi_gpu(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly when using multiple GPUs
    """
    model = BoringModel()
    trainer = Trainer(
        gpus=2,
        plugins='fully_sharded',
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)

    _assert_save_equality(tmpdir, trainer)


def _assert_save_equality(tmpdir, trainer):
    if trainer.global_rank == 0:

        checkpoint_path = os.path.join(tmpdir, 'model.pt')
        trainer.save_checkpoint(checkpoint_path)
        saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

        # Ensure we gather all shards for comparison
        model_state_dict = trainer.accelerator.training_type_plugin.collate_state_dict()
        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)
