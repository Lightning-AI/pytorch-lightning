# Copyright The PyTorch Lightning team.
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
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_8
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

if torch.distributed.is_available() and _TORCH_GREATER_EQUAL_1_8:
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD


@RunIf(skip_windows=True, min_torch="1.9.0", min_gpus=2, special=True)
def test_ddp_fp16_compress_comm_hook(tmpdir):
    """Test for DDP FP16 compress hook."""
    model = BoringModel()
    training_type_plugin = DDPPlugin(
        ddp_comm_hook=default.fp16_compress_hook,
        sync_batchnorm=True,
    )
    trainer = Trainer(
        max_epochs=1,
        gpus=2,
        plugins=[training_type_plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
        fast_dev_run=True,
    )
    trainer.fit(model)
    trainer_comm_hook = (trainer.accelerator.training_type_plugin._model.get_ddp_logging_data().comm_hook)
    expected_comm_hook = default.fp16_compress_hook.__qualname__
    assert trainer_comm_hook == expected_comm_hook
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(skip_windows=True, min_torch="1.9.0", min_gpus=2, special=True)
def test_ddp_sgd_comm_hook(tmpdir):
    """Test for DDP FP16 compress hook."""
    model = BoringModel()
    training_type_plugin = DDPPlugin(
        ddp_comm_state=powerSGD.PowerSGDState(process_group=None),
        ddp_comm_hook=powerSGD.powerSGD_hook,
        sync_batchnorm=True,
    )
    trainer = Trainer(
        max_epochs=1,
        gpus=2,
        plugins=[training_type_plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
        fast_dev_run=True,
    )
    trainer.fit(model)
    trainer_comm_hook = (trainer.accelerator.training_type_plugin._model.get_ddp_logging_data().comm_hook)
    expected_comm_hook = powerSGD.powerSGD_hook.__qualname__
    assert trainer_comm_hook == expected_comm_hook
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(skip_windows=True, min_torch="1.9.0", min_gpus=2, special=True)
def test_ddp_fp16_compress_wrap_sgd_comm_hook(tmpdir):
    """Test for DDP FP16 compress wrapper for SGD hook."""
    model = BoringModel()
    training_type_plugin = DDPPlugin(
        ddp_comm_state=powerSGD.PowerSGDState(process_group=None),
        ddp_comm_hook=powerSGD.powerSGD_hook,
        ddp_comm_wrapper=default.fp16_compress_wrapper,
        sync_batchnorm=True,
    )
    trainer = Trainer(
        max_epochs=1,
        gpus=2,
        plugins=[training_type_plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
        fast_dev_run=True,
    )
    trainer.fit(model)
    trainer_comm_hook = (trainer.accelerator.training_type_plugin._model.get_ddp_logging_data().comm_hook)
    expected_comm_hook = default.fp16_compress_wrapper(powerSGD.powerSGD_hook).__qualname__
    assert trainer_comm_hook == expected_comm_hook
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(skip_windows=True, min_torch="1.9.0", min_gpus=2, special=True)
def test_ddp_spawn_fp16_compress_comm_hook(tmpdir):
    """Test for DDP Spawn FP16 compress hook."""
    model = BoringModel()
    training_type_plugin = DDPSpawnPlugin(
        ddp_comm_hook=default.fp16_compress_hook,
        sync_batchnorm=True,
    )
    trainer = Trainer(
        max_epochs=1,
        gpus=2,
        plugins=[training_type_plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
        fast_dev_run=True,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
