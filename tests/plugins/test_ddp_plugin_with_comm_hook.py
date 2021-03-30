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
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.plugins import DDPPlugin
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_7
if _TORCH_GREATER_EQUAL_1_7:
    from torch.distributed.algorithms.ddp_comm_hooks import (
        default_hooks as default,
        powerSGD_hook as powerSGD,
    )


class BoringHalfGrdadientPrecisionCheckModel(BoringModel):

    def on_after_backward(self):
        super().on_after_backward()
        for k, v in self.named_parameters():
            assert v.grad.dtype == torch.half

@RunIf(skip_windows=True, min_torch="1.7.0", min_gpus=2)
def test_ddp_fp16_compress_comm_hook(tmpdir):
    """Test for DDP FP16 compress hook."""
    model = BoringHalfGrdadientPrecisionCheckModel()
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
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"

@RunIf(skip_windows=True, min_torch="1.9.0", min_gpus=2)
def test_ddp_fp16_compress_wrap_sgd_comm_hook(tmpdir):
    """Test for DDP FP16 compress wrapper for SGD hook."""
    model = BoringHalfGrdadientPrecisionCheckModel()
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
    )
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
