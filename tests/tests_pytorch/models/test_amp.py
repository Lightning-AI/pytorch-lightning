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
from unittest import mock

import pytest
import torch
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from torch.utils.data import DataLoader

import tests_pytorch.helpers.utils as tutils
from tests_pytorch.helpers.runif import RunIf


class AMPTestModel(BoringModel):
    def step(self, batch):
        self._assert_autocast_enabled()
        output = self(batch)
        is_bfloat16 = self.trainer.precision_plugin.precision == "bf16-mixed"
        assert output.dtype == torch.float16 if not is_bfloat16 else torch.bfloat16
        return self.loss(output)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._assert_autocast_enabled()
        output = self(batch)
        is_bfloat16 = self.trainer.precision_plugin.precision == "bf16-mixed"
        assert output.dtype == torch.float16 if not is_bfloat16 else torch.bfloat16
        return output

    def _assert_autocast_enabled(self):
        if self.trainer.precision_plugin.device == "cpu":
            assert torch.is_autocast_cpu_enabled()
        else:
            assert torch.is_autocast_enabled()


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("strategy", "precision", "devices"),
    [
        ("single_device", "16-mixed", 1),
        ("single_device", "bf16-mixed", 1),
        ("ddp_spawn", "16-mixed", 2),
        pytest.param("ddp_spawn", "bf16-mixed", 2, marks=RunIf(skip_windows=True)),
    ],
)
def test_amp_cpus(tmp_path, strategy, precision, devices):
    """Make sure combinations of AMP and strategies work if supported."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = AMPTestModel()
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)


@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
@pytest.mark.parametrize(
    "devices", [pytest.param(1, marks=RunIf(min_cuda_gpus=1)), pytest.param(2, marks=RunIf(min_cuda_gpus=2))]
)
def test_amp_gpus(tmp_path, precision, devices):
    """Make sure combinations of AMP and strategies work if supported."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        accelerator="gpu",
        devices=devices,
        strategy=("ddp_spawn" if devices > 1 else "auto"),
        precision=precision,
    )

    model = AMPTestModel()
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model, DataLoader(RandomDataset(32, 64)))


@RunIf(min_cuda_gpus=1)
@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "1",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
        "SLURM_PROCID": "0",
    },
)
def test_amp_gpu_ddp_slurm_managed(tmp_path):
    """Make sure DDP + AMP work."""
    # simulate setting slurm flags
    model = AMPTestModel()

    # exp file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        accelerator="gpu",
        devices=[0],
        strategy="ddp_spawn",
        precision="16-mixed",
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)
    assert isinstance(trainer.strategy.cluster_environment, SLURMEnvironment)


@pytest.mark.parametrize("clip_val", [0, 10])
@mock.patch("torch.nn.utils.clip_grad_norm_")
def test_precision_16_clip_gradients(mock_clip_grad_norm, clip_val, tmp_path):
    """Ensure that clip gradients is only called if the value is greater than 0."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        devices=1,
        precision="16-mixed",
        limit_train_batches=4,
        limit_val_batches=0,
        gradient_clip_val=clip_val,
    )
    trainer.fit(model)

    if clip_val > 0:
        mock_clip_grad_norm.assert_called()
    else:
        mock_clip_grad_norm.assert_not_called()
