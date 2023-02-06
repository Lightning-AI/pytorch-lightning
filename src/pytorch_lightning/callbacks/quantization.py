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
r"""
Quantization
^^^^^^^^^^^^

"""
import copy
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.ao.quantization.qconfig import QConfig
from torch.quantization import FakeQuantizeBase

import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_11, _TORCH_GREATER_EQUAL_1_12
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_11:
    from torch.ao.quantization import fuse_modules_qat as fuse_modules
else:
    from torch.quantization import fuse_modules


def wrap_qat_forward_context(
    quant_cb: "QuantizationAwareTraining",
    model: "pl.LightningModule",
    func: Callable,
    trigger_condition: Optional[Union[Callable, int]] = None,
) -> Callable:
    """Decorator to wrap forward path as it is needed to quantize inputs and dequantize outputs for in/out
    compatibility Moreover this version has the (de)quantization conditional as it may not be needed for the
    training all the time."""
    # todo: consider using registering hook before/after forward
    @functools.wraps(func)
    def wrapper(data: Any) -> Any:
        _is_func_true = callable(trigger_condition) and trigger_condition(model.trainer)
        _is_count_true = isinstance(trigger_condition, int) and quant_cb._forward_calls < trigger_condition
        _quant_run = trigger_condition is None or _is_func_true or _is_count_true
        # apply custom trigger
        if _quant_run:
            quant_cb._forward_calls += 1
            data = model.quant(data)  # type: ignore[operator]
        data = func(data)
        # apply custom trigger
        if _quant_run:
            data = model.dequant(data)  # type: ignore[operator]
        return data

    return wrapper


def wrap_quantize_forward_context(model: "pl.LightningModule", func: Callable) -> Callable:
    """Decorator to wrap forward path as it is needed to quantize inputs and dequantize outputs for in/out
    compatibility."""
    # todo: consider using registering hook before/after forward
    @functools.wraps(func)
    def wrapper(data: Any) -> Any:
        data = model.quant(data)  # type: ignore[operator]
        data = func(data)
        data = model.dequant(data)  # type: ignore[operator]
        return data

    return wrapper


def _recursive_hasattr(obj: Any, attribs: str, state: bool = True) -> bool:
    """recursive check if model has some layers denoted with '.'."""
    if "." in attribs:
        attrib, attribs = attribs.split(".", 1)
        if hasattr(obj, attrib):
            return _recursive_hasattr(getattr(obj, attrib), attribs, state)
        return False
    return state and hasattr(obj, attribs)


class QuantizationAwareTraining(Callback):
    """Quantization allows speeding up inference and decreasing memory requirements by performing computations and
    storing tensors at lower bitwidths (such as INT8 or FLOAT16) than floating point precision. We use native
    PyTorch API so for more information see `PyTorch Quantization`_.

    .. warning:: ``QuantizationAwareTraining`` is in beta and subject to change.

    The ``LightningModule`` is prepared for QAT training in the ``on_fit_start`` hook. Checkpoints saved during training
    include already collected stats to perform the Quantization conversion, but it doesn't contain the quantized or
    fused model/layers. The quantization is performed in the ``on_fit_end`` hook so the model needs to be saved after
    training finishes if quantization is desired.

    Args:

        qconfig: quantization configuration:

            - 'fbgemm' for server inference.
            - 'qnnpack' for mobile inference.
            - a custom `torch.quantization.QConfig`_.

        observer_type: allows switching between ``MovingAverageMinMaxObserver`` as "average" (default)
            and ``HistogramObserver`` as "histogram" which is more computationally expensive.

        collect_quantization: count or custom function to collect quantization statistics:

            - ``None`` (default). The quantization observer is called in each module forward
                (useful for collecting extended statistic when using image/data augmentation).
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
            but break compatibility to torchscript and export with ``torch.save``.

        quantize_on_fit_end: perform the quantization in `on_fit_end`.
            Note that once converted, the model cannot be put in training mode again.

        observer_enabled_stages: allow fake-quantization modules' observers to do calibration during provided stages:

            - ``'train'``: the observers can do calibration during training.
            - ``'validate'``: the observers can do calibration during validating.
              Note that we don't disable observers during the sanity check as the model hasn't been calibrated with
              training data yet. After the sanity check, the fake-quantization modules are restored to initial states.
            - ``'test'``: the observers can do calibration during testing.
            - ``'predict'``: the observers can do calibration during predicting.

            Note that we only handle observers belonging to fake-quantization modules. When ``qconfig`` is a ``str`` and
            ``observer_type`` is ``'histogram'``, the observers won't belong to any fake-quantization modules and will
            not be controlled by the callback.

    .. _PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html#quantization-aware-training
    .. _torch.quantization.QConfig:
        https://pytorch.org/docs/stable/generated/torch.quantization.qconfig.QConfig.html#qconfig
    """

    OBSERVER_TYPES = ("histogram", "average")
    OBSERVER_STAGES = ("train", "validate", "test", "predict")

    def __init__(
        self,
        qconfig: Union[str, QConfig] = "fbgemm",
        observer_type: str = "average",
        collect_quantization: Optional[Union[int, Callable]] = None,
        modules_to_fuse: Optional[Sequence] = None,
        input_compatible: bool = True,
        quantize_on_fit_end: bool = True,
        observer_enabled_stages: Sequence[str] = ("train",),
    ) -> None:
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
        self._observer_type = observer_type

        if collect_quantization is not None and not (
            isinstance(collect_quantization, int) or callable(collect_quantization)
        ):
            raise MisconfigurationException(
                f'Unsupported `collect_quantization` "{collect_quantization}", allowed are `int` or `Callable`.'
            )
        self._collect_quantization = collect_quantization

        self._modules_to_fuse = modules_to_fuse
        self._input_compatible = input_compatible
        self._convert_on_fit_end = quantize_on_fit_end

        observer_enabled_stages = set(observer_enabled_stages)
        unsupported_stages = observer_enabled_stages - set(self.OBSERVER_STAGES)
        if unsupported_stages:
            raise MisconfigurationException(
                f'Unsupported stages "{tuple(sorted(unsupported_stages))}", allowed are {self.OBSERVER_STAGES}.'
            )
        self._observer_disabled_stages = set(self.OBSERVER_STAGES) - observer_enabled_stages

        self._forward_calls = 0
        self._fake_quant_to_initial_state_dict: Dict[FakeQuantizeBase, Dict[str, Any]] = {}
        self._last_fake_quant_to_observer_enabled: Dict[FakeQuantizeBase, Tensor] = {}
        self._module_prepared = False

    def _check_feasible_fuse(self, model: "pl.LightningModule") -> bool:
        if not self._modules_to_fuse:
            return False
        for group in self._modules_to_fuse:
            if not all(_recursive_hasattr(model, m) for m in group):
                raise MisconfigurationException(
                    f"You have requested to fuse {group} but one or more of them is not your model attributes"
                )
        return True

    def _collect_observer_enabled(self) -> Dict[FakeQuantizeBase, Tensor]:
        return {
            fake_quant: fake_quant.observer_enabled.clone() for fake_quant in self._fake_quant_to_initial_state_dict
        }

    def _disable_observer(self, pl_module: "pl.LightningModule") -> None:
        self._last_fake_quant_to_observer_enabled = self._collect_observer_enabled()
        pl_module.apply(torch.quantization.disable_observer)

    def _restore_last_observer_enabled(self) -> None:
        for fake_quant, observer_enabled in self._last_fake_quant_to_observer_enabled.items():
            fake_quant.observer_enabled.copy_(observer_enabled)

    def _prepare_model(self, model: "pl.LightningModule") -> None:
        if self._module_prepared:
            return
        # QuantStub converts tensors from floating point to quantized
        model.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        model.dequant = torch.quantization.DeQuantStub()
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        self.__module_forward = model.forward
        model.forward = wrap_qat_forward_context(  # type: ignore [assignment]
            quant_cb=self, model=model, func=model.forward, trigger_condition=self._collect_quantization
        )

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference
        if isinstance(self._qconfig, str):
            if self._observer_type == "histogram":
                model.qconfig = torch.quantization.get_default_qconfig(self._qconfig)
            elif self._observer_type == "average":
                model.qconfig = torch.quantization.get_default_qat_qconfig(
                    self._qconfig, version=0 if _TORCH_GREATER_EQUAL_1_12 else None
                )

        elif isinstance(self._qconfig, QConfig):
            model.qconfig = self._qconfig  # type: ignore [assignment]

        if self._check_feasible_fuse(model):
            fuse_modules(model, self._modules_to_fuse, inplace=True)

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        torch.quantization.prepare_qat(model, inplace=True)

        fake_quants = tuple(module for module in model.modules() if isinstance(module, FakeQuantizeBase))
        self._fake_quant_to_initial_state_dict = {
            fake_quant: copy.deepcopy(fake_quant.state_dict()) for fake_quant in fake_quants
        }
        self._module_prepared = True

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._prepare_model(pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._convert_on_fit_end:
            pl_module.forward = self.__module_forward  # type: ignore [assignment]
            return
        pl_module.eval()
        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        torch.quantization.convert(pl_module, inplace=True)
        # check we shall preserve wrapper
        if self._input_compatible:
            pl_module.forward = wrap_quantize_forward_context(  # type: ignore [assignment]
                model=pl_module,
                func=self.__module_forward,
            )
        else:
            pl_module.forward = self.__module_forward  # type: ignore [assignment]

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "train" in self._observer_disabled_stages:
            self._disable_observer(pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "train" in self._observer_disabled_stages:
            self._restore_last_observer_enabled()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "validate" in self._observer_disabled_stages and not trainer.sanity_checking:
            # ``torch.quantization.MovingAveragePerChannelMinMaxObserver`` and ``torch.quantization.HistogramObserver``
            # need to see at least one batch to infer the shapes of quantization ``scale`` and ``zero_point``. So we
            # don't disable observers during the sanity check so that they can infer the shapes of quantization
            # parameters with validation data.
            self._disable_observer(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "validate" in self._observer_disabled_stages:
            if trainer.sanity_checking:
                for fake_quant, state_dict in self._fake_quant_to_initial_state_dict.items():
                    fake_quant.load_state_dict(state_dict)
            else:
                self._restore_last_observer_enabled()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "test" in self._observer_disabled_stages:
            self._disable_observer(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "test" in self._observer_disabled_stages:
            self._restore_last_observer_enabled()

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "predict" in self._observer_disabled_stages:
            self._disable_observer(pl_module)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "predict" in self._observer_disabled_stages:
            self._restore_last_observer_enabled()

    def state_dict(self) -> Dict[str, Any]:
        keys = {"_qconfig", "_observer_type", "_collect_quantization", "_modules_to_fuse", "_input_compatible"}
        return {n: getattr(self, n) for n in keys}

    def _load_before_model(self, model: "pl.LightningModule", state_dict: Dict[str, Any]) -> None:
        """Special hook that gets called by the CheckpointConnector *before* the model gets loaded.

        This hook replaces the :meth:`on_load_checkpoint` and :meth:`load_state_dict` callback methods which get called
        after the model has already loaded the weights. For quantization, we need to convert the model first before that
        happens, assuming the previous training used quantization.
        """
        for k, v in state_dict.items():
            setattr(self, k, v)
        self._prepare_model(model)
