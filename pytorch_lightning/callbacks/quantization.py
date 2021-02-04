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
from typing import Any, Callable

import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.model_helpers import is_overridden


def wrap_quantize_forward_context(func: Callable) -> Callable:
    """
    This decorator is used as context manager...
    """

    def wrapper(self, batch) -> Any:
        batch = self.quant(batch)
        batch = self.forward(batch)
        batch = self.dequant(batch)
        return batch

    return wrapper


class StaticModelQuantization(Callback):

    def __init__(
        self,
            num_batches: int = 3,
    ) -> None:
        self.num_batches = num_batches
        self._count_validations = 0
        self.qmodel = None

    def on_fit_start(self, trainer, pl_module):
        # specify which part to quantize and how
        pl_module.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        # todo: fuse selected by user or write func to fuse all
        # torch.quantization.fuse_modules(pl_module, modules_to_fuse=..., inplace=True)

        torch.quantization.prepare(pl_module, inplace=True)
        # self.qmodel = torch.quantization.prepare(pl_module, inplace=False)
        # self.qmodel.eval()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._count_validations += 1

    def on_fit_end(self, trainer, pl_module):
        # change code directly or use manipulation APIs
        # pl_module.eval()

        # # todo: if missing validation
        # for idx, batch in enumerate(trainer.train_dataloader):
        #     if idx >= self.num_batches:
        #         break
        #     self.qmodel(batch)
        #
        # # convert to quantized version
        # torch.quantization.convert(self.qmodel, inplace=True)

        pass


class QuantizationAwareTraining(Callback):

    def __init__(
        self,
    ) -> None:
        pass

    def on_fit_start(self, trainer, pl_module):

        pl_module.quant = torch.quantization.QuantStub()
        pl_module.dequant = torch.quantization.DeQuantStub()
        pl_module.forward = wrap_quantize_forward_context(pl_module.forward)

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference
        pl_module.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        torch.quantization.prepare_qat(pl_module, inplace=True)

    def on_fit_end(self, trainer, pl_module):
        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        pl_module.eval()
        self.qmodel = torch.quantization.convert(pl_module)
