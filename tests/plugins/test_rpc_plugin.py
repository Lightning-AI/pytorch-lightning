import os
from typing import Optional
from unittest import mock

import pytest
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.rpc_plugin import RPCPlugin
from pytorch_lightning.utilities import RPC_AVAILABLE
from tests.base.boring_model import BoringModel


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
@pytest.mark.skipif(not RPC_AVAILABLE, reason="RPC is not available")
def test_rpc_choice(tmpdir, ddp_backend, gpus, num_processes):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, RPCPlugin)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        distributed_backend=ddp_backend,
        callbacks=[CB()],
        plugins=[RPCPlugin()]
    )

    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)


class CustomRPCPlugin(RPCPlugin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rpc_save_model_count = 0
        self.on_main_rpc_connect_count = 0
        self.worker_optimizer_step_count = 0
        self.is_main_rpc_process_count = 0
        self.on_exit_rpc_process_count = 0
        self.return_after_exit_rpc_process_count = 0

    def on_accelerator_exit_rpc_process(self, trainer) -> None:
        self.on_exit_rpc_process_count += 1

    def rpc_save_model(self, save_model_fn, last_filepath, trainer, pl_module) -> None:
        self.rpc_save_model_count += 1

    def on_main_rpc_connection(self, trainer) -> None:
        self.on_main_rpc_connect_count += 1

    def worker_optimizer_step(self, model: LightningModule, opt_idx: int, *args, **kwargs) -> None:
        self.worker_optimizer_step_count += 1

    @property
    def is_main_rpc_process(self) -> bool:
        self.is_main_rpc_process_count += 1
        return torch.distributed.get_rank() == 0

    @property
    def return_after_exit_rpc_process(self) -> bool:
        self.return_after_exit_rpc_process_count += 1
        return False

    def barrier(self, name: Optional[str] = None) -> None:
        return


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not RPC_AVAILABLE, reason="RPC is not available")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
def test_rpc_function_calls_ddp(tmpdir):
    model = BoringModel()
    plugin = CustomRPCPlugin()
    max_epochs = 2
    limit_train_batches = 2
    trainer = Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=max_epochs,
        gpus=2,
        distributed_backend='ddp',
        plugins=[plugin]
    )

    trainer.fit(model)
    if trainer.global_rank == 0:  # Main process
        assert plugin.rpc_save_model_count == max_epochs
        assert plugin.on_main_rpc_connect_count == 1
        assert plugin.worker_optimizer_step_count == max_epochs * limit_train_batches
        # Call once at init, and at optim step
        assert plugin.is_main_rpc_process_count == 1 + plugin.worker_optimizer_step_count
        assert plugin.on_exit_rpc_process_count == 0
    else:  # Worker process
        assert plugin.rpc_save_model_count == max_epochs
        assert plugin.on_main_rpc_connect_count == 0
        # Never signaled by worker, only by main process
        assert plugin.worker_optimizer_step_count == 0
        # Call once at init, and at optim step
        assert plugin.is_main_rpc_process_count == 1 + (max_epochs * limit_train_batches)
        # Called at init
        assert plugin.on_exit_rpc_process_count == 1
