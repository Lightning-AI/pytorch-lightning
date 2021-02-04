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
from typing import Any, Callable, Union

import torch
from torch.quantization import MinMaxObserver, QConfig

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


def wrap_quantize_forward_context(model, func: Callable) -> Callable:
    """
    This decorator is used as context manager...
    """

    def wrapper(*args, **kwargs) -> Any:
        # todo: not clear argument passing
        res = model.quant(*args, **kwargs)
        res = func(res)
        res = model.dequant(res)
        return res

    return wrapper


class QuantizationAwareTraining(Callback):

    def __init__(
        self,
        qconfig: Union[str, QConfig] = 'fbgemm',
        fusing = None,
    ) -> None:
        if not isinstance(qconfig, (str, QConfig)):
            raise MisconfigurationException(f"Unsupported qconfig: f{self._qconfig}")
        self._qconfig = qconfig

    def on_fit_start(self, trainer, pl_module):

        pl_module.quant = torch.quantization.QuantStub()
        pl_module.dequant = torch.quantization.DeQuantStub()
        pl_module.forward = wrap_quantize_forward_context(pl_module, pl_module.forward)

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference
        if isinstance(self._qconfig, str):
            pl_module.qconfig = torch.quantization.get_default_qat_qconfig(self._qconfig)
        elif isinstance(self._qconfig, QConfig):
            pl_module.qconfig = self._qconfig

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        torch.quantization.prepare_qat(pl_module, inplace=True)

    def on_fit_end(self, trainer, pl_module):
        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        pl_module.eval()
        torch.quantization.convert(pl_module, inplace=True)
