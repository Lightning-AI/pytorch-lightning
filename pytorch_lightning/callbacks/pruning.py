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
Pruning
^^^^^^^

Perform model pruning.

"""
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

from torch import nn
from torch.nn.modules.container import ModuleDict, ModuleList

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import _PYTORCH_PRUNE_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _PYTORCH_PRUNE_AVAILABLE:
    import torch.nn.utils.prune as pytorch_prune


_PYTORCH_PRUNING_FUNCTIONS = {
    "ln_structured": pytorch_prune.ln_structured,
    "l1_unstructured": pytorch_prune.l1_unstructured,
    "random_structured": pytorch_prune.random_structured,
    "random_unstructured": pytorch_prune.random_unstructured,
}

_PYTORCH_PRUNING_METHOD = {
    "ln_structured": pytorch_prune.LnStructured,
    "l1_unstructured": pytorch_prune.L1Unstructured,
    "random_structured": pytorch_prune.RandomStructured,
    "random_unstructured": pytorch_prune.RandomUnstructured,
}


def check_parameters_to_prune(
    pl_module: LightningModule,
    parameters_to_prune: Optional[List],
    parameters: List[str] = ["weight"]
) -> List:

    is_parameters_to_prune_none = parameters_to_prune is None
    current_modules = [m for m in pl_module.modules() if not isinstance(m, (LightningModule, ModuleDict, ModuleList))]

    if parameters_to_prune is None:
        parameters_to_prune = []
        for p in parameters:
            for m in current_modules:
                param = getattr(m, p, None)
                if param is not None:
                    parameters_to_prune.append((m, p))

    if isinstance(parameters_to_prune, list) and not is_parameters_to_prune_none:

        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in parameters_to_prune):

            missing_modules = []
            missing_parameters = []

            for module, param_name in parameters_to_prune:
                if module not in current_modules:
                    missing_modules.append(module)
                    continue

                parameter = getattr(module, param_name)

                if parameter is None:
                    missing_parameters.append(parameter)

            if len(missing_modules) > 0 or len(missing_parameters) > 0:
                raise MisconfigurationException(
                    "Ths provided parameters_to_tune doesn't exist in the model. "
                    f"Found mismatching modules: {missing_modules} and missing_parameters: {missing_parameters}"
                )

    return parameters_to_prune


class PruningCallback(Callback):

    PARAMETER_NAMES = ["weight", "bias"]

    def __init__(
        self,
        pruning_fn: Union[Callable, str],
        parameters_to_prune: Optional[List[Union[str, Tuple[nn.Module, str]]]] = None,
        parameter_names: List[str] = ["weight"],
        use_global_unstructured: bool = True,
        amount: Optional[Union[int, float]] = 0.5,
        prune_on_epoch_end: Optional[bool] = False,
        prune_on_fit_end: Optional[bool] = True,
        make_pruning_permanent: Optional[bool] = True,
        use_lottery_ticket_hypothesis: Optional[bool] = True,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
    ) -> None:
        """

        Pruning Callback relying on PyTorch prune utils.

        This callback is responsible to prune networks parameters
        during your training.

        Args:

            pruning_fn: function from torch.nn.utils.prune module
                or your based on BasePruningMethod. Can be string e.g.
                `"l1_unstructured"`. See pytorch docs for more details.

            parameters_to_prune: list of strings or list of tuple with
                nn.Module and its associated string name parameters.

            parameter_names: List of parameter names to be used from nn.Module.
                Can either be `weight` or `bias`.

            use_global_unstructured: Whether to apply pruning globally on the model.
                If parameters_to_prune is provided, global_unstructured will be restricted on them.

            amount: quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and
                represent the fraction of parameters to prune.
                If int, it represents the absolute number
                of parameters to prune.
                If Callable, the function will be called on every epoch.

            prune_on_epoch_end: bool flag determines call or not
                to call pruning_fn on epoch end.

            prune_on_fit_end: bool flag determines call or not
                to call pruning_fn on fit end.

            make_pruning_permanent: if True then all
                reparametrization pre-hooks and tensors with mask
                will be removed on fit end.

            use_lottery_ticket_hypothesis: Wether to use algorithm describes in
                "The lottery ticket hypothesis" (https://arxiv.org/pdf/1803.03635.pdf)

            pruning_dim: if you are using structured pruning method you need
                to specify dimension.

            pruning_norm: if you are using ln_structured you need to specify norm.
        """

        self.use_global_unstructured = use_global_unstructured
        self.parameters_to_prune = parameters_to_prune
        self.prune_on_epoch_end = prune_on_epoch_end
        self.prune_on_fit_end = prune_on_fit_end
        self.use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis
        self.parameter_names = parameter_names or self.PARAMETER_NAMES

        for param_name in self.parameter_names:
            if param_name not in self.PARAMETER_NAMES:
                raise MisconfigurationException(
                    f"The provided parameter_names {param_name} isn't in {self.PARAMETER_NAMES} "
                )

        if isinstance(pruning_fn, str):
            if pruning_fn not in _PYTORCH_PRUNING_FUNCTIONS.keys():
                raise MisconfigurationException(
                    f"The provided pruning_fn {pruning_fn} isn't available with "
                    f"PyTorch build-in {_PYTORCH_PRUNING_FUNCTIONS.keys()} "
                )
            if "unstructured" not in pruning_fn:
                if pruning_dim is None:
                    raise MisconfigurationException(
                        "When requesting `structured` pruning, the `pruning_dim` should be provided."
                    )
                if pruning_fn == "ln_structured":
                    if pruning_norm is None:
                        raise MisconfigurationException(
                            "When requesting `ln_structured` pruning, the `pruning_norm` should be provided."
                        )

                    pruning_fn = self.create_pruning_fn(pruning_fn, dim=pruning_dim, n=pruning_norm)
            else:
                pruning_fn = self.create_pruning_fn(pruning_fn)

        self.pruning_fn = pruning_fn

        if not (prune_on_epoch_end or prune_on_fit_end):
            rank_zero_warn(
                "The PruningCallback won't be triggered as not activate either on epoch_en or fit_end.", UserWarning)

        self.make_pruning_permanent = make_pruning_permanent

        if not isinstance(amount, (int, float, Callable)):
            raise MisconfigurationException(
                "amount should be provided and be either an int, a float or Callable function."
            )

        self.amount = amount

    def create_pruning_fn(self, pruning_fn: str, *args, **kwargs):
        if self.use_global_unstructured:
            pruning_fn = _PYTORCH_PRUNING_METHOD[pruning_fn]
            self._global_kwargs = kwargs
            return pruning_fn
        else:
            return PruningCallback._wrap_pruning_fn(_PYTORCH_PRUNING_FUNCTIONS[pruning_fn], **kwargs)

    @staticmethod
    def _wrap_pruning_fn(pruning_fn, *args, **kwargs):
        return lambda module, name, amount: pruning_fn(
            module, name, amount, *args, **kwargs
        )

    def _make_pruning_permanent(self):
        for module, param_name in self.parameters_to_prune:
            pytorch_prune.remove(module, param_name)

    def _resolve_amout(self, current_epoch: int) -> float:
        if isinstance(self.amount, Callable):
            amount_fn = self.amount
            amount = amount_fn(current_epoch)
        else:
            amount = self.amount
        return amount

    def _apply_lottery_ticket_hypothesis(self, module: nn.Module, orig_module: nn.Module, tensor_name: str):
        trained = getattr(module, tensor_name)
        orig = getattr(orig_module, tensor_name)
        trained.data = orig.data.to(trained.device)

    def apply_lottery_ticket_hypothesis(self):
        for (module, tensor_name), (orig_module, _) in zip(self.parameters_to_prune, self._parameters_to_prune):
            self._apply_lottery_ticket_hypothesis(module, orig_module, tensor_name)

    def _apply_local_pruning(self, amount: float):
        for module, param in self.parameters_to_prune:
            self.pruning_fn(module, name=param, amount=amount)

    def _apply_global_pruning(self, amount: float):
        pytorch_prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=self.pruning_fn,
            amount=amount,
            **self._global_kwargs
        )

    def apply_pruning(self, trainer, pl_module):
        amount = self._resolve_amout(trainer.current_epoch)
        if self.use_global_unstructured:
            self._apply_global_pruning(amount)
        else:
            self._apply_local_pruning(amount)

        if self.use_lottery_ticket_hypothesis:
            self.apply_lottery_ticket_hypothesis()

    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        self.parameters_to_prune = check_parameters_to_prune(
            pl_module, self.parameters_to_prune, parameters=self.parameter_names)

        if self.use_lottery_ticket_hypothesis:
            # make a copy of copy of orginal weights.
            self._parameters_to_prune = [(deepcopy(m), n) for m, n in self.parameters_to_prune]

    def on_epoch_end(self, trainer, pl_module):
        self.apply_pruning(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        self.apply_pruning(trainer, pl_module)

        if self.make_pruning_permanent:
            self._make_pruning_permanent()
