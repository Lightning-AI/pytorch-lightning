# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0

import os
import time
from pathlib import Path

import pytest
import torch

from lightning.fabric.plugins.io.distributed_async_io import DistributedAsyncCheckpointIO
from lightning.pytorch import Trainer
from lightning.pytorch.demos import BoringModel
from tests_pytorch.helpers.runif import RunIf

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _sync_across_ranks(trainer, obj):
    """Broadcast an object from rank 0 once the strategy/process group exists."""
    trainer.strategy.barrier()
    obj = trainer.strategy.broadcast(obj, src=0)
    trainer.strategy.barrier()
    return obj


def _wait_for_dcp_metadata(path: Path, timeout: int = 10):
    """DCP writes metadata last; wait until it appears."""
    start = time.time()
    while True:
        if any(p.name.startswith(".metadata") for p in path.iterdir()):
            return
        if time.time() - start > timeout:
            raise RuntimeError("Checkpoint metadata not visible yet")
        time.sleep(0.1)


def _find_checkpoint(tmp_path: Path):
    """Poll until a checkpoint file exists (async IO may delay visibility)."""
    ckpt_dir = tmp_path / "lightning_logs" / "version_0" / "checkpoints"

    for _ in range(100):
        files = list(ckpt_dir.glob("*.ckpt"))
        if files:
            return max(files, key=os.path.getctime)
        time.sleep(0.1)

    raise RuntimeError(f"Checkpoint file not found in {ckpt_dir}")


# -------------------------------------------------------------------------
# Core logic
# -------------------------------------------------------------------------


def save_model_checkpoint(tmp_path, expected_strategy_name, accelerator, devices):
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=10,
        devices=devices,
        accelerator=accelerator,
        plugins=[DistributedAsyncCheckpointIO()],
    )

    assert trainer.strategy.__class__.__name__ == expected_strategy_name, (
        f"Expected strategy {expected_strategy_name}, but got {trainer.strategy.__class__.__name__}"
    )

    trainer.fit(model)

    # Important:
    # pytest standalone gives each worker a different tmp_path.
    # After DDP init (fit), broadcast rank0's path so all ranks agree.
    tmp_path = _sync_across_ranks(trainer, tmp_path)

    return tmp_path  # noqa: RET504


def load_model_checkpoint(tmp_path, expected_strategy_name, accelerator, devices):
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=20,
        devices=devices,
        accelerator=accelerator,
        plugins=[DistributedAsyncCheckpointIO()],
    )

    assert trainer.strategy.__class__.__name__ == expected_strategy_name, (
        f"Expected strategy {expected_strategy_name}, but got {trainer.strategy.__class__.__name__}"
    )

    last_ckpt = _find_checkpoint(Path(tmp_path))

    trainer.fit(model, ckpt_path=last_ckpt)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


@RunIf(min_torch="2.4", min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    ("expected_strategy_name", "devices"),
    [
        ("SingleDeviceStrategy", 1),
        ("DDPStrategy", 2),
    ],
)
def test_trainer_distributed_async_checkpointio_integration_cuda(tmp_path, expected_strategy_name, devices):
    torch.manual_seed(1234)

    tmp_path = save_model_checkpoint(
        tmp_path,
        expected_strategy_name,
        accelerator="cuda",
        devices=devices,
    )

    ckpt_path = _find_checkpoint(Path(tmp_path))
    _wait_for_dcp_metadata(ckpt_path)

    load_model_checkpoint(
        tmp_path,
        expected_strategy_name,
        accelerator="cuda",
        devices=devices,
    )


@RunIf(min_torch="2.4", standalone=True)
def test_trainer_distributed_async_checkpointio_integration_cpu(tmp_path):
    torch.manual_seed(1234)

    save_model_checkpoint(
        tmp_path,
        "SingleDeviceStrategy",
        accelerator="cpu",
        devices=1,
    )

    ckpt_path = _find_checkpoint(Path(tmp_path))
    _wait_for_dcp_metadata(ckpt_path)

    load_model_checkpoint(
        tmp_path,
        "SingleDeviceStrategy",
        accelerator="cpu",
        devices=1,
    )
