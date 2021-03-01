import os
import platform

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDPShardedPlugin, DDPSpawnShardedPlugin
from pytorch_lightning.utilities import _APEX_AVAILABLE, _FAIRSCALE_AVAILABLE, _NATIVE_AMP_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


@pytest.mark.parametrize(["accelerator"], [("ddp_sharded", ), ("ddp_sharded_spawn", )])
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_sharded_ddp_choice(tmpdir, accelerator):
    """
        Test to ensure that plugin is correctly chosen
    """

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            if accelerator == 'ddp_sharded':
                assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)
            elif accelerator == 'ddp_sharded_spawn':
                assert isinstance(trainer.accelerator.training_type_plugin, DDPSpawnShardedPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator=accelerator,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(not _APEX_AVAILABLE, reason="test requires apex")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_invalid_apex_sharded(tmpdir):
    """
        Test to ensure that we raise an error when we try to use apex and sharded
    """

    model = BoringModel()
    with pytest.raises(MisconfigurationException, match='Sharded Plugin is not supported with Apex AMP'):
        trainer = Trainer(
            fast_dev_run=True,
            accelerator='ddp_sharded_spawn',
            precision=16,
            amp_backend='apex',
        )

        trainer.fit(model)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires GPU machine")
@pytest.mark.parametrize(["accelerator"], [("ddp_sharded", ), ("ddp_sharded_spawn", )])
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="Requires native AMP")
def test_ddp_choice_sharded_amp(tmpdir, accelerator):
    """
        Test to ensure that plugin native amp plugin is correctly chosen when using sharded
    """

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            if accelerator == 'ddp_sharded':
                assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)
            elif accelerator == 'ddp_sharded_spawn':
                assert isinstance(trainer.accelerator.training_type_plugin, DDPSpawnShardedPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=1,
        precision=16,
        accelerator=accelerator,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_checkpoint_cpu(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        num_processes=2,
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for ddp_param, shard_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(ddp_param.to("cpu"), shard_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_checkpoint_multi_gpu(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly when using multiple GPUs
    """
    model = BoringModel()
    trainer = Trainer(
        gpus=2,
        accelerator='ddp_sharded_spawn',
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for ddp_param, shard_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(ddp_param.to("cpu"), shard_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_finetune(tmpdir):
    """
        Test to ensure that we can save and restart training (simulate fine-tuning)
    """
    model = BoringModel()
    trainer = Trainer(
        gpus=2,
        accelerator='ddp_sharded_spawn',
        fast_dev_run=True,
    )
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(fast_dev_run=True, )
    trainer.fit(saved_model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint(tmpdir):
    """
        Test to ensure that resuming from checkpoint works
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        num_processes=2,
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        num_processes=2,
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path,
    )

    trainer.fit(model)


@pytest.mark.skip(reason="Not a critical test, skip till drone CI performance improves.")
@pytest.mark.skip(reason="Currently unsupported restarting training on different number of devices.")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint_downsize_gpus(tmpdir):
    """
        Test to ensure that resuming from checkpoint works when downsizing number of GPUS
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        fast_dev_run=True,
        gpus=2,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        fast_dev_run=True,
        gpus=1,
        resume_from_checkpoint=checkpoint_path,
    )

    trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint_gpu_to_cpu(tmpdir):
    """
        Test to ensure that resuming from checkpoint works when going from GPUs- > CPU
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        gpus=1,
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        num_processes=2,
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path,
    )

    trainer.fit(model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
def test_ddp_sharded_plugin_test(tmpdir):
    """
        Test to ensure we can use test without fit
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        num_processes=2,
        fast_dev_run=True,
    )

    trainer.test(model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not _FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_test_multigpu(tmpdir):
    """
        Test to ensure we can use test without fit
    """
    model = BoringModel()
    trainer = Trainer(
        accelerator='ddp_sharded_spawn',
        gpus=2,
        fast_dev_run=True,
    )

    trainer.test(model)
