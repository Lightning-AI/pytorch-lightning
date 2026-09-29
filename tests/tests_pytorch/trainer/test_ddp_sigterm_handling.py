import os
import signal
import time

import pytest
import torch
import torch.multiprocessing as mp

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.demos.boring_classes import BoringDataModule
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.exceptions import SIGTERMException

# Skip the test if DDP or multiple devices are not available

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or torch.cuda.device_count() < 2,
    reason="Requires torch.distributed and at least 2 CUDA devices",
)


class DummyModel(LightningModule):
    def training_step(self, batch, batch_idx):
        # Simulate SIGTERM in rank 0 at batch 2
        if self.trainer.global_rank == 0 and batch_idx == 2:
            time.sleep(3)  # Let other ranks proceed to the next batch
            os.kill(os.getpid(), signal.SIGTERM)
        return super().training_step(batch, batch_idx)


def run_ddp_sigterm(rank, world_size, tmpdir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    seed_everything(42)

    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    model = DummyModel()
    datamodule = BoringDataModule()

    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=world_size,
        num_nodes=1,
        max_epochs=3,
        default_root_dir=tmpdir,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
    except SIGTERMException:
        # Test passed: SIGTERM was properly raised and caught
        print(f"[Rank {rank}] Caught SIGTERMException successfully.")
    except Exception as e:
        pytest.fail(f"[Rank {rank}] Unexpected exception: {e}")


def test_ddp_sigterm_handling(tmp_path):
    world_size = 2
    mp.spawn(run_ddp_sigterm, args=(world_size, tmp_path), nprocs=world_size, join=True)


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="Requires torch.distributed",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.device_count() < 2,
    reason="Requires >=2 CUDA devices or use CPU",
)
def test_sigterm_handling_ddp(tmp_path):
    test_ddp_sigterm_handling(tmp_path)
