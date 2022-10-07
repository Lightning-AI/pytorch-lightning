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
from pytorch_lightning.demos.boring_classes import BoringModel


class _BoringModelForEnableGrad(BoringModel):
    def on_test_epoch_start(self) -> None:
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        return super().on_test_epoch_start()


class _BoringModelForNoGrad(BoringModel):
    def on_test_epoch_start(self) -> None:
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled
        return super().on_test_epoch_start()


def test_inference_grad_mode():
    """Testing overwriting trainer arguments."""
    trainer = Trainer(logger=False, inference_grad_mode=True)
    trainer.test(_BoringModelForEnableGrad())
    trainer = Trainer(logger=False, inference_grad_mode=False)
    trainer.test(_BoringModelForNoGrad())
 