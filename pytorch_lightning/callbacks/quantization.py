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
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch.quantization import QConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def wrap_qat_forward_context(
    quant_cb,
    model: pl.core.LightningModule,
    func: Callable,
    trigger_condition: Optional[Union[Callable, int]] = None
) -> Callable:
    """
    This decorator is used as context manager...
    """

    # todo: consider using registering hook before/after forward
    def wrapper(data) -> Any:
        _is_func_true = isinstance(trigger_condition, Callable) and trigger_condition(model.trainer)
        _is_count_true = isinstance(trigger_condition, int) and quant_cb._forward_calls < trigger_condition
        _quant_run = trigger_condition is None or _is_func_true or _is_count_true
        # apply custom trigger
        if _quant_run:
            quant_cb._forward_calls += 1
            data = model.quant(data)
        data = func(data)
        # apply custom trigger
        if _quant_run:
            data = model.dequant(data)
        return data

    return wrapper


def wrap_quantize_forward_context(model: pl.core.LightningModule, func: Callable, quant, dequant) -> Callable:
    """
    This decorator is used as context manager...
    """

    # todo: consider using registering hook before/after forward
    def wrapper(data) -> Any:
        data = model.quant(data)
        data = func(data)
        data = model.dequant(data)
        return data

    return wrapper


def _recursive_hasattr(obj, attribs, state=True):
    """recursive check if model has some layers denoted with ."""
    if '.' in attribs:
        attrib, attribs = attribs.split('.', 1)
        if hasattr(obj, attrib):
            return _recursive_hasattr(getattr(obj, attrib), attribs, state)
        else:
            return False
    else:
        return state & hasattr(obj, attribs)


class QuantizationAwareTraining(Callback):
    """
    Quantization allows speed-up inference and decrease model size by performing computations and storing tensors
     at lower bitwidths than floating point precision such as INT8 or FLOAT16.
    We use native PyTorch API so for more information see
     `Quantization <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>_`
    """

    OBSERVER_TYPES = (None, 'histogram', 'average')

    def __init__(
        self,
        qconfig: Union[str, QConfig] = 'fbgemm',
        observer_type: Optional[str] = None,
        collect_quantization: Optional[Union[int, Callable]] = None,
        modules_to_fuse: Optional[Sequence] = None,
        input_compatible: bool = True,
    ) -> None:
        """
        Args:
            qconfig: define quantization configuration see: `torch.quantization.QConfig
             <https://pytorch.org/docs/stable/torch.quantization.html?highlight=qconfig#torch.quantization.QConfig>_`
                or use pre-defined: 'fbgemm' for server inference and 'qnnpack' for mobile inference
            observer_type: switching between `MovingAverageMinMaxObserver` as "average"
                and `HistogramObserver` as "histogram" which is more computational costly
            collect_quantization: define custom count or function when you shall collect quantization statistic

                - with default ``None`` you call the quantization observer for each module forward,
                    typical use-case can be reflecting user augmentation
                - custom call count to limit explicit number of calls starting from beginning
                - custom lambda function with single trainer argument,
                    see example when you limit call only for last epoch::

                    def custom_trigger_last(trainer):
                        return trainer.current_epoch == (trainer.max_epochs - 1)

                    QuantizationAwareTraining(lambda_trigger=custom_trigger_last)

            modules_to_fuse: allows you fuse a few layer together as you can see in `diagram
             <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>_`
                to find what layer types can be fue=sed check https://github.com/pytorch/pytorch/pull/43286
            input_compatible: preserve quent/dequant layers which allows to feat any input as to the original model,
                but break compatibility to torchscript
        """
        if not isinstance(qconfig, (str, QConfig)):
            raise MisconfigurationException(f"Unsupported qconfig: f{qconfig}.")
        self._qconfig = qconfig

        if observer_type not in self.OBSERVER_TYPES:
            raise MisconfigurationException(
                f'Unsupported observer type "{observer_type}", allowed are {self.OBSERVER_TYPES}.'
            )
        self._observer_type = observer_type

        if collect_quantization is not None and not isinstance(collect_quantization, (int, Callable)):
            raise MisconfigurationException(
                f'Unsupported `collect_quantization` "{observer_type}", allowed are int or callable.'
            )
        self._collect_quantization = collect_quantization

        self.modules_to_fuse = modules_to_fuse
        self._preserve_compatible = input_compatible
        self._forward_calls = 0

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
        self.__module_forward = pl_module.forward
        pl_module.forward = wrap_qat_forward_context(
            quant_cb=self, model=pl_module, func=pl_module.forward, trigger_condition=self._collect_quantization
        )

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference
        if isinstance(self._qconfig, str):
            if self._observer_type == 'average':
                pl_module.qconfig = torch.quantization.get_default_qat_qconfig(self._qconfig)
            else:
                pl_module.qconfig = torch.quantization.get_default_qconfig(self._qconfig)
        elif isinstance(self._qconfig, QConfig):
            pl_module.qconfig = self._qconfig

        if self.modules_to_fuse:
            self._check_feasible_fuse(pl_module)
            torch.quantization.fuse_modules(pl_module, self.modules_to_fuse, inplace=True)

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        torch.quantization.prepare_qat(pl_module, inplace=True)

    def on_fit_end(self, trainer, pl_module):
        pl_module.eval()
        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        torch.quantization.convert(pl_module, inplace=True)
        # check we shall preserve wrapper
        if self._preserve_compatible:
            pl_module.forward = wrap_quantize_forward_context(
                model=pl_module, func=self.__module_forward, quant=pl_module.quant, dequant=pl_module.dequant
            )
        else:
            pl_module.forward = self.__module_forward
