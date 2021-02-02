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

r"""
Quantization
^^^^^^^^^^^^

"""
import torch

from pytorch_lightning.callbacks.base import Callback


class StaticModelQuantization(Callback):

    def __init__(
        self,
    ) -> None:
        self.num_batches = 3

    def on_fit_end(self, trainer, pl_module):
        # tweak model for best results
        # change code directly or use manipulation APIs
        pl_module.eval()
        # pl_module = torch.quantization.fuse_modules(pl_module, [["conv1", "bn1", "relu1"]])

        # specify which part to quantize and how
        pl_module.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        # configurable!

        qmodel = torch.quantization.prepare(pl_module, inplace=False)

        # collect calibration statistics
        qmodel.eval()
        for idx, batch in enumerate(trainer.train_dataloader):
            if idx >= self.num_batches:
                break
            pl_module.validation_step(self, batch, idx)

        # convert to quantized version
        torch.quantization.convert(qmodel, inplace=True)


class QuantizationAwareTraining(Callback):

    def __init__(
        self,
    ) -> None:
        pass

    # todo
