# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from pathlib import Path

import pytest
import torch

from lightning.fabric.plugins.io.distributed_async_io import DistributedAsyncCheckpointIO
from lightning.pytorch import Trainer
from lightning.pytorch.demos import BoringModel
from tests_pytorch.helpers.runif import RunIf

# --- integration test to verify the checkpoint is actually saved and loaded asynchronously ---


def _wait_for_dcp_metadata(path: Path, timeout=10):
    # writing files in CI can be slow,
    # and DCP writes a metadata file last,
    # so we can wait for that to appear to ensure the checkpoint is ready
    start = time.time()
    while True:
        # DCP metadata file pattern
        if any(p.name.startswith(".metadata") for p in path.iterdir()):
            return
        if time.time() - start > timeout:
            raise RuntimeError("Checkpoint metadata not visible yet")
        time.sleep(0.1)


def save_model_checkpoint(tmp_path, expected_strategy_name, accelerator, devices):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=10,
        devices=devices,
        plugins=[DistributedAsyncCheckpointIO()],
        accelerator=accelerator,
    )
    assert trainer.strategy.__class__.__name__ == expected_strategy_name, (
        f"Expected strategy {expected_strategy_name}, but got {trainer.strategy.__class__.__name__}"
    )
    trainer.fit(model)


def get_checkpoint_path(tmp_path):
    tmp_path = Path(tmp_path)
    ckpt_path = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_files = list(ckpt_path.glob("*.ckpt"))
    assert len(ckpt_files) > 0, "No checkpoint files found"
    return max(ckpt_files, key=os.path.getctime)


def load_model_checkpoint(tmp_path, expected_strategy_name, accelerator, devices):
    last_ckpt = get_checkpoint_path(tmp_path)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=20,
        devices=devices,
        plugins=[DistributedAsyncCheckpointIO()],
        accelerator=accelerator,
    )
    assert trainer.strategy.__class__.__name__ == expected_strategy_name, (
        f"Expected strategy {expected_strategy_name}, but got {trainer.strategy.__class__.__name__}"
    )

    trainer.fit(model, ckpt_path=last_ckpt)  # if loading works, it will restore to epoch 10 and continue to 20


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
    save_model_checkpoint(tmp_path, expected_strategy_name, accelerator="cuda", devices=devices)

    ckpt_path = get_checkpoint_path(tmp_path)
    _wait_for_dcp_metadata(ckpt_path)

    load_model_checkpoint(tmp_path, expected_strategy_name, accelerator="cuda", devices=devices)


@RunIf(min_torch="2.4", standalone=True)
def test_trainer_distributed_async_checkpointio_integration_cpu(tmp_path):
    torch.manual_seed(1234)
    save_model_checkpoint(tmp_path, "SingleDeviceStrategy", accelerator="cpu", devices=1)

    ckpt_path = get_checkpoint_path(tmp_path)
    _wait_for_dcp_metadata(ckpt_path)

    load_model_checkpoint(tmp_path, "SingleDeviceStrategy", accelerator="cpu", devices=1)
