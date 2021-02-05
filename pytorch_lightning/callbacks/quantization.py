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
from typing import Any, Callable, Union, Optional, Sequence

import torch
from torch.quantization import MinMaxObserver, QConfig

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


def wrap_quantize_forward_context(model, func: Callable) -> Callable:
    """
    This decorator is used as context manager...
    """

    def wrapper(data) -> Any:
        # todo: consider using registering hook before/after forward
        data = model.quant(data)
        data = func(data)
        data = model.dequant(data)
        return data

    return wrapper


def _recursive_hasattr(obj, attribs, state=True):
    if '.' in attribs:
        attrib, attribs = attribs.split('.', 1)
        if hasattr(obj, attrib):
            return _recursive_hasattr(getattr(obj, attrib), attribs, state)
        else:
            return False
    else:
        return state & hasattr(obj, attribs)


class QuantizationAwareTraining(Callback):

    def __init__(
        self,
        qconfig: Union[str, QConfig] = 'fbgemm',
        modules_to_fuse: Optional[Sequence] = None,  # https://github.com/pytorch/pytorch/pull/43286
    ) -> None:
        if not isinstance(qconfig, (str, QConfig)):
            raise MisconfigurationException(f"Unsupported qconfig: f{self._qconfig}")
        self._qconfig = qconfig
        self.modules_to_fuse = modules_to_fuse

    def _check_feasible_fuse(self, model):
        if not self.modules_to_fuse:
            return True
        for group in self.modules_to_fuse:
            if not all(_recursive_hasattr(model, m) for m in group):
                raise MisconfigurationException(
                    f'You have requested to fuse {group} but one or more of them is not your model attributes'
                )
        return True

    def on_fit_start(self, trainer, pl_module):
        # QuantStub converts tensors from floating point to quantized
        pl_module.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        pl_module.dequant = torch.quantization.DeQuantStub()
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        pl_module.forward = wrap_quantize_forward_context(pl_module, pl_module.forward)

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference
        if isinstance(self._qconfig, str):
            pl_module.qconfig = torch.quantization.get_default_qat_qconfig(self._qconfig)
        elif isinstance(self._qconfig, QConfig):
            pl_module.qconfig = self._qconfig

        if self.modules_to_fuse:
            self._check_feasible_fuse(pl_module)
            torch.quantization.fuse_modules(pl_module, self.modules_to_fuse, inplace=True)

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

# OP_LIST_TO_FUSER_METHOD = {
#     (torch.nn.Conv1d, torch.nn.BatchNorm1d): fuse_conv_bn,
#     (torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU): fuse_conv_bn_relu,
#     (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn,
#     (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_conv_bn_relu,
#     (torch.nn.Conv3d, torch.nn.BatchNorm3d): fuse_conv_bn,
#     (torch.nn.Conv3d, torch.nn.BatchNorm3d, torch.nn.ReLU): fuse_conv_bn_relu,
#     (torch.nn.Conv1d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU1d,
#     (torch.nn.Conv2d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU2d,
#     (torch.nn.Conv3d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU3d,
#     (torch.nn.Linear, torch.nn.ReLU): torch.nn.intrinsic.LinearReLU,
#     (torch.nn.BatchNorm2d, torch.nn.ReLU): torch.nn.intrinsic.BNReLU2d,
#     (torch.nn.BatchNorm3d, torch.nn.ReLU): torch.nn.intrinsic.BNReLU3d,
# }
