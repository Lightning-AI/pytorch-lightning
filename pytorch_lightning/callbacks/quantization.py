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
import functools
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch.quantization import QConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_LOWER_EQUAL_1_4


def wrap_qat_forward_context(
    quant_cb,
    model: pl.core.LightningModule,
    func: Callable,
    trigger_condition: Optional[Union[Callable, int]] = None
) -> Callable:
    """
    Decorator to wrap forward path as it is needed to quantize inputs and dequantize outputs for in/out compatibility
    Moreover this version has the (de)quantization conditional as it may not be needed for the training all the time
    """
    # todo: consider using registering hook before/after forward
    @functools.wraps(func)
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


def wrap_quantize_forward_context(model: pl.core.LightningModule, func: Callable) -> Callable:
    """
    Decorator to wrap forward path as it is needed to quantize inputs and dequantize outputs for in/out compatibility
    """
    # todo: consider using registering hook before/after forward
    @functools.wraps(func)
    def wrapper(data) -> Any:
        data = model.quant(data)
        data = func(data)
        data = model.dequant(data)
        return data

    return wrapper


def _recursive_hasattr(obj: Any, attribs: str, state: bool = True) -> bool:
    """recursive check if model has some layers denoted with '.'"""
    if '.' in attribs:
        attrib, attribs = attribs.split('.', 1)
        if hasattr(obj, attrib):
            return _recursive_hasattr(getattr(obj, attrib), attribs, state)
        return False
    return state and hasattr(obj, attribs)


class QuantizationAwareTraining(Callback):
    OBSERVER_TYPES = ('histogram', 'average')

    def __init__(
        self,
        qconfig: Union[str, QConfig] = 'fbgemm',
        observer_type: str = "average",
        collect_quantization: Optional[Union[int, Callable]] = None,
        modules_to_fuse: Optional[Sequence] = None,
        input_compatible: bool = True,
    ) -> None:
        """
        Quantization allows speeding up inference and decreasing memory requirements
        by performing computations and storing tensors at lower bitwidths
        (such as INT8 or FLOAT16) than floating point precision.
        We use native PyTorch API so for more information
        see `Quantization <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>`_.

        .. warning:: ``QuantizationAwareTraining`` is in beta and subject to change.


        Args:

            qconfig: quantization configuration:

                - 'fbgemm' for server inference.
                - 'qnnpack' for mobile inference.
                -  a custom `torch.quantization.QConfig <https://pytorch.org/docs/stable/torch.quantization.html#torch.quantization.QConfig>`_.

            observer_type: allows switching between ``MovingAverageMinMaxObserver`` as "average" (default)
                and ``HistogramObserver`` as "histogram" which is more computationally expensive.

            collect_quantization: count or custom function to collect quantization statistics:

                - ``None`` (deafult). The quantization observer is called in each module forward
                    (useful for collecting extended statistic when useing image/data augmentation).
                - ``int``. Use to set a fixed number of calls, starting from the beginning.
                - ``Callable``. Custom function with single trainer argument.
                    See this example to trigger only the last epoch:

                    .. code-block:: python

                        def custom_trigger_last(trainer):
                            return trainer.current_epoch == (trainer.max_epochs - 1)

                        QuantizationAwareTraining(collect_quantization=custom_trigger_last)

            modules_to_fuse: allows you fuse a few layers together as shown in
                `diagram <https://pytorch.org/docs/stable/quantization.html#quantization-aware-training>`_
                to find which layer types can be fused, check https://github.com/pytorch/pytorch/pull/43286.

            input_compatible: preserve quant/dequant layers. This allows to feat any input as to the original model,
                but break compatibility to torchscript.

        """  # noqa: E501
        _valid_qconf_str = isinstance(qconfig, str) and qconfig in torch.backends.quantized.supported_engines
        if not isinstance(qconfig, QConfig) and not _valid_qconf_str:
            raise MisconfigurationException(
                f"Unsupported qconfig: f{qconfig}.\nTry one of defaults: {torch.backends.quantized.supported_engines}"
            )
        self._qconfig = qconfig

        if observer_type not in self.OBSERVER_TYPES:
            raise MisconfigurationException(
                f'Unsupported observer type "{observer_type}", allowed are {self.OBSERVER_TYPES}.'
            )
        elif observer_type == 'histogram' and _TORCH_LOWER_EQUAL_1_4:
            raise MisconfigurationException(f'For using {observer_type} you need to be using pytorch>=1.5.')
        self._observer_type = observer_type

        if collect_quantization is not None and not isinstance(collect_quantization, (int, Callable)):
            raise MisconfigurationException(
                f'Unsupported `collect_quantization` "{collect_quantization}", allowed are `int` or `Callable`.'
            )
        self._collect_quantization = collect_quantization

        self.modules_to_fuse = modules_to_fuse
        self._input_compatible = input_compatible
        self._forward_calls = 0

    def _check_feasible_fuse(self, model):
        if not self.modules_to_fuse:
            return False
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
            if self._observer_type == 'histogram':
                pl_module.qconfig = torch.quantization.get_default_qconfig(self._qconfig)
            elif self._observer_type == 'average':
                pl_module.qconfig = torch.quantization.get_default_qat_qconfig(self._qconfig)
        elif isinstance(self._qconfig, QConfig):
            pl_module.qconfig = self._qconfig

        if self._check_feasible_fuse(pl_module):
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
        if self._input_compatible:
            pl_module.forward = wrap_quantize_forward_context(model=pl_module, func=self.__module_forward)
        else:
            pl_module.forward = self.__module_forward
