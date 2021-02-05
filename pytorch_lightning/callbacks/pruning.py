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
ModelPruning
^^^^^^^^^^^^

"""
import inspect
from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch.nn.utils.prune as pytorch_prune
from torch import nn
from torch.nn.modules.container import ModuleDict, ModuleList

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

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


class ModelPruning(Callback):
    PARAMETER_NAMES = ("weight", "bias")

    def __init__(
        self,
        pruning_fn: Optional[Union[Callable, str]] = None,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        parameter_names: Optional[List[str]] = None,
        use_global_unstructured: bool = True,
        amount: Union[int, float, Callable] = 0.5,
        make_pruning_permanent: bool = True,
        use_lottery_ticket_hypothesis: bool = True,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
    ) -> None:
        """

        Pruning Callback relying on PyTorch prune utils.

        This callback is responsible to prune networks parameters
        during your training.

        Find here the PyTorch (Pruning Tutorial)[https://pytorch.org/tutorials/intermediate/pruning_tutorial.html]

        .. code-block:: python

            parameters_to_prune = [
                (model.mlp_1, "weight"),
                (model.mlp_2, "weight")
            ]

            trainer = Trainer(callbacks=[
                ModelPruning(
                    pruning_fn='l1_unstructured',
                    parameters_to_prune=parameters_to_prune,
                    amount=0.01,
                    use_global_unstructured=True,
                )
            ])

        When `parameters_to_prune` is None, `parameters_to_prune` will contains all parameters from the model.
        The user can override `filter_parameters_to_prune` to filter any nn.Module to be pruned.

        Args:

            pruning_fn: function from torch.nn.utils.prune module
                or your based own subclasses from PyTorch ``BasePruningMethod``.
                Can be string e.g. `"l1_unstructured"`.
                See pytorch docs for more details.

            parameters_to_prune: list of strings or list of tuple with
                nn.Module and its associated string name parameters.

            parameter_names: List of parameter names to be used from nn.Module.
                Can either be `weight` or `bias`.

            use_global_unstructured: Whether to apply pruning globally on the model.
                If parameters_to_prune is provided, global_unstructured will be restricted on them.

            amount: quantity of parameters to prune:

                - float, between 0.0 and 1.0. represents the fraction of parameters to prune.
                - int, represents the absolute number of parameters to prune.
                - Callable, for dynamic values. will be called every epoch.

            make_pruning_permanent: if True then all reparametrization pre-hooks and tensors
                with mask will be removed on fit end.

            use_lottery_ticket_hypothesis: Whether to use algorithm describes in
                "The lottery ticket hypothesis" (https://arxiv.org/pdf/1803.03635.pdf)

            pruning_dim: if you are using structured pruning method you need
                to specify dimension.

            pruning_norm: if you are using ln_structured you need to specify norm.

        """

        self._use_global_unstructured = use_global_unstructured
        self._parameters_to_prune = parameters_to_prune
        self._use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis
        self._parameter_names = parameter_names or self.PARAMETER_NAMES
        self._global_kwargs = {}
        self._initial_parameters_to_prune = None

        for param_name in self._parameter_names:
            if param_name not in self.PARAMETER_NAMES:
                raise MisconfigurationException(
                    f"The provided `parameter_names`: {param_name} isn't in {self.PARAMETER_NAMES}"
                )

        if isinstance(pruning_fn, str):
            pruning_kwargs = {}
            pruning_fn = pruning_fn.lower()
            if pruning_fn not in _PYTORCH_PRUNING_FUNCTIONS:
                raise MisconfigurationException(
                    f"The provided `pruning_fn` {pruning_fn} isn't available in PyTorch's"
                    f" built-in functions: {list(_PYTORCH_PRUNING_FUNCTIONS.keys())} "
                )
            if pruning_fn.endswith("_structured"):
                if pruning_dim is None:
                    raise MisconfigurationException(
                        "When requesting `structured` pruning, the `pruning_dim` should be provided."
                    )
                if pruning_fn == "ln_structured":
                    if pruning_norm is None:
                        raise MisconfigurationException(
                            "When requesting `ln_structured` pruning, the `pruning_norm` should be provided."
                        )
                    pruning_kwargs["n"] = pruning_norm
                pruning_kwargs["dim"] = pruning_dim
            pruning_fn = self._create_pruning_fn(pruning_fn, **pruning_kwargs)
        elif self.is_pruning_method(pruning_fn):
            if not use_global_unstructured:
                # TODO: currently not supported
                raise MisconfigurationException(
                    "PyTorch `BasePruningMethod` is currently only supported with `use_global_unstructured=True`."
                )
        else:
            raise MisconfigurationException(
                f"`pruning_fn` is expected to be a str in {list(_PYTORCH_PRUNING_FUNCTIONS.keys())}"
                f" or a PyTorch `BasePruningMethod`. Found: {pruning_fn}."
                " HINT: if passing a `BasePruningMethod`, pass the the class, not an instance"
            )

        if use_global_unstructured and pruning_fn.PRUNING_TYPE != "unstructured":
            raise MisconfigurationException(
                'Only the "unstructured" PRUNING_TYPE is supported with `use_global_unstructured=True`.'
                f" Found method {pruning_fn} of type {pruning_fn.PRUNING_TYPE}. "
            )

        self.pruning_fn = pruning_fn
        self.make_pruning_permanent = make_pruning_permanent

        if not isinstance(amount, (int, float, Callable)):
            raise MisconfigurationException(
                "amount should be provided and be either an int, a float or Callable function."
            )

        self.amount = amount

    def filter_parameters_to_prune(self, parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None):
        """
        This function can be overridden to control which module to prune.
        """
        return parameters_to_prune

    def _create_pruning_fn(self, pruning_fn: str, **kwargs):
        """
        This function takes `pruning_fn`, a function name.

        IF use_global_unstructured, pruning_fn will be resolved into its associated ``PyTorch BasePruningMethod``
        ELSE, pruning_fn will be resolved into its function counterpart from `torch.nn.utils.prune`.

        """
        if self._use_global_unstructured:
            pruning_fn = _PYTORCH_PRUNING_METHOD[pruning_fn]
            self._global_kwargs = kwargs
            return pruning_fn
        return ModelPruning._wrap_pruning_fn(_PYTORCH_PRUNING_FUNCTIONS[pruning_fn], **kwargs)

    @staticmethod
    def _wrap_pruning_fn(pruning_fn, **kwargs):
        return partial(pruning_fn, **kwargs)

    def _make_pruning_permanent(self):
        for module, param_name in self._parameters_to_prune:
            pytorch_prune.remove(module, param_name)

    def _resolve_amount(self, current_epoch: int) -> float:
        return self.amount(current_epoch) if isinstance(self.amount, Callable) else self.amount

    def _restore_original_weights(self, module: nn.Module, orig_module: nn.Module, tensor_name: str):
        """
        "The lottery ticket hypothesis" (https://arxiv.org/pdf/1803.03635.pdf) algorithm:

            1. Randomly initialize a neural network f(x;θ0)(where θ0 ∼Dθ).
            2. Train the network for j iterations, arriving at parameters θj .
            3. Prune p% of the parameters in θj , creating a mask m.
            4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; m⊙θ0).

        This function is responsible of step 4.
        """
        trained = getattr(module, tensor_name)
        orig = getattr(orig_module, tensor_name)
        if trained is None or orig is None:
            return
        trained.data = orig.data.to(trained.device)

    def apply_lottery_ticket_hypothesis(self):
        for (mod, tensor_name), (initial_mod, _) in zip(self._parameters_to_prune, self._initial_parameters_to_prune):
            self._restore_original_weights(mod, initial_mod, tensor_name)

    def _apply_local_pruning(self, amount: float):
        for module, param in self._parameters_to_prune:
            self.pruning_fn(module, name=param, amount=amount)

    def _resolve_global_kwargs(self, amount: float):
        self._global_kwargs["amount"] = amount
        params = set(inspect.signature(self.pruning_fn).parameters)
        params.discard("self")
        return {k: v for k, v in self._global_kwargs.items() if k in params}

    def _apply_global_pruning(self, amount: float):
        pytorch_prune.global_unstructured(
            self._parameters_to_prune, pruning_method=self.pruning_fn, **self._resolve_global_kwargs(amount)
        )

    def apply_pruning(self, current_epoch: int):
        amount = self._resolve_amount(current_epoch)

        # the user could control the pruning frequency with amount_fn
        if not amount:
            return

        if self._use_global_unstructured:
            self._apply_global_pruning(amount)
        else:
            self._apply_local_pruning(amount)

        if self._use_lottery_ticket_hypothesis:
            self.apply_lottery_ticket_hypothesis()

    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        parameters_to_prune = self.sanitize_parameters_to_prune(
            pl_module, self._parameters_to_prune, parameters=self._parameter_names
        )

        self._parameters_to_prune = self.filter_parameters_to_prune(parameters_to_prune)

        if self._use_lottery_ticket_hypothesis:
            # make a copy of copy of original weights.
            self._initial_parameters_to_prune = [(deepcopy(m), n) for m, n in self._parameters_to_prune]

    def on_epoch_end(self, trainer, pl_module):
        self.apply_pruning(trainer.current_epoch)

        if self.make_pruning_permanent:
            self._make_pruning_permanent()

    @staticmethod
    def sanitize_parameters_to_prune(
        pl_module: LightningModule,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        parameters: Optional[List[str]] = None,
    ) -> List[Tuple[nn.Module, str]]:
        """
        This function is responsible to check provided ``parameters_to_prune` and `parameters`.
        If parameters_to_prune is None, parameters_to_prune will be generated from all parameters of the model.
        """
        parameters = parameters or ModelPruning.PARAMETER_NAMES

        current_modules = [
            m for m in pl_module.modules() if not isinstance(m, (LightningModule, ModuleDict, ModuleList))
        ]

        if parameters_to_prune is None:
            parameters_to_prune = [(m, p) for p in parameters for m in current_modules if hasattr(m, p)]
        elif (
            isinstance(parameters_to_prune, (list, tuple))
            and len(parameters_to_prune) > 0
            and all(len(p) == 2 for p in parameters_to_prune)
            and all(isinstance(a, nn.Module) and isinstance(b, str) for a, b in parameters_to_prune)
        ):
            missing_modules, missing_parameters = [], []
            for module, param_name in parameters_to_prune:
                if module not in current_modules:
                    missing_modules.append(module)
                    continue
                if not hasattr(module, param_name):
                    missing_parameters.append(param_name)

            if len(missing_modules) > 0 or len(missing_parameters) > 0:
                raise MisconfigurationException(
                    "Some provided `parameters_to_tune` don't exist in the model."
                    f" Found missing modules: {missing_modules} and missing parameters: {missing_parameters}"
                )
        else:
            raise MisconfigurationException(
                "The provided `parameters_to_prune` should either be list of tuple "
                "with 2 elements: (nn.Module in your model, parameter_name_to_prune) or None"
            )

        return parameters_to_prune

    @staticmethod
    def is_pruning_method(method) -> bool:
        if not inspect.isclass(method):
            return False
        return issubclass(method, pytorch_prune.BasePruningMethod)
