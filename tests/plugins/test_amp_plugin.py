from pytorch_lightning.callbacks import Callback
from tests.base.boring_model import BoringModel
from pytorch_lightning import Trainer
import pytest
import os
from unittest import mock
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from distutils.version import LooseVersion
import torch


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6",
)
@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "SLURM_NTASKS": "2",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(['ddp_backend', 'gpus', 'num_processes'],
                         [('ddp_cpu', None, None), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)])
def test_amp_choice_default_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_connector.backend, NativeAMPPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='native',
        gpus=gpus,
        num_processes=num_processes,
        distributed_backend=ddp_backend,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6",
)
@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "SLURM_NTASKS": "2",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(['ddp_backend', 'gpus', 'num_processes'],
                         [('ddp_cpu', None, None), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)])
def test_amp_choice_custom_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    class MyNativeAMP(NativeAMPPlugin):
        pass

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_connector.backend, MyNativeAMP)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='native',
        gpus=gpus,
        num_processes=num_processes,
        distributed_backend=ddp_backend,
        plugins=[MyNativeAMP()],
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)
