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
Finetuning Callback
^^^^^^^^^^^^^^^^^^^^
Freeze and unfreeze models for finetuning purposes
"""
import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)


def multiplicative(epoch: int) -> float:
    return 2.0


class BaseFinetuning(Callback):
    r"""
    This class implements the base logic for writing your own Finetuning Callback.

    Override ``freeze_before_training`` and ``finetune_function`` methods with your own logic.

    ``freeze_before_training``: This method is called before ``configure_optimizers``
        and should be used to freeze any modules parameters.

    ``finetune_function``: This method is called on every train epoch start and should be used to
        ``unfreeze`` any parameters. Those parameters needs to be added in a new ``param_group``
        within the optimizer.

    .. note:: Make sure to filter the parameters based on ``requires_grad``.

    Example::

        >>> from torch.optim import Adam
        >>> class MyModel(pl.LightningModule):
        ...     def configure_optimizer(self):
        ...         # Make sure to filter the parameters based on `requires_grad`
        ...         return Adam(filter(lambda p: p.requires_grad, self.parameters()))
        ...
        >>> class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
        ...     def __init__(self, unfreeze_at_epoch=10):
        ...         super().__init__()
        ...         self._unfreeze_at_epoch = unfreeze_at_epoch
        ...
        ...     def freeze_before_training(self, pl_module):
        ...         # freeze any module you want
        ...         # Here, we are freezing `feature_extractor`
        ...         self.freeze(pl_module.feature_extractor)
        ...
        ...     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        ...         # When `current_epoch` is 10, feature_extractor will start training.
        ...         if current_epoch == self._unfreeze_at_epoch:
        ...             self.unfreeze_and_add_param_group(
        ...                 modules=pl_module.feature_extractor,
        ...                 optimizer=optimizer,
        ...                 train_bn=True,
        ...             )
    """

    def __init__(self) -> None:
        self._internal_optimizer_metadata: Dict[int, List[Dict[str, Any]]] = {}
        self._restarting = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._restarting = True
        if "internal_optimizer_metadata" in state_dict:
            self._internal_optimizer_metadata = state_dict["internal_optimizer_metadata"]
        else:
            # compatibility to load from old checkpoints before PR #11887
            self._internal_optimizer_metadata = state_dict  # type: ignore[assignment]

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # restore the param_groups created during the previous training.
        if self._restarting:
            named_parameters = dict(pl_module.named_parameters())
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                param_groups = self._apply_mapping_to_param_groups(
                    self._internal_optimizer_metadata[opt_idx], named_parameters
                )
                optimizer.param_groups = param_groups
            self._restarting = False

    @staticmethod
    def flatten_modules(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> List[Module]:
        """This function is used to flatten a module or an iterable of modules into a list of its leaf modules
        (modules with no children) and parent modules that have parameters directly themselves.

        Args:
            modules: A given module or an iterable of modules

        Returns:
            List of modules
        """
        if isinstance(modules, ModuleDict):
            modules = modules.values()

        if isinstance(modules, Iterable):
            _flatten_modules = []
            for m in modules:
                _flatten_modules.extend(BaseFinetuning.flatten_modules(m))

            _modules = iter(_flatten_modules)
        else:
            _modules = modules.modules()

        # Capture all leaf modules as well as parent modules that have parameters directly themselves
        return [m for m in _modules if not list(m.children()) or m._parameters]

    @staticmethod
    def filter_params(
        modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True, requires_grad: bool = True
    ) -> Generator:
        """Yields the `requires_grad` parameters of a given module or list of modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: Whether not to train the BatchNorm module
            requires_grad: Whether to create a generator for trainable or non-trainable parameters.
        Returns:
            Generator
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and not train_bn:
                continue
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in mod.parameters(recurse=False):
                if param.requires_grad == requires_grad:
                    yield param

    @staticmethod
    def make_trainable(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        """Unfreezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for module in modules:
            if isinstance(module, _BatchNorm):
                module.track_running_stats = True
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in module.parameters(recurse=False):
                param.requires_grad = True

    @staticmethod
    def freeze_module(module: Module) -> None:
        """Freezes the parameters of the provided module.

        Args:
            module: A given module
        """
        if isinstance(module, _BatchNorm):
            module.track_running_stats = False
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = False

    @staticmethod
    def freeze(modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode

        Returns:
            None
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(mod)
            else:
                BaseFinetuning.freeze_module(mod)

    @staticmethod
    def filter_on_optimizer(optimizer: Optimizer, params: Iterable) -> List:
        """This function is used to exclude any parameter which already exists in this optimizer.

        Args:
            optimizer: Optimizer used for parameter exclusion
            params: Iterable of parameters used to check against the provided optimizer

        Returns:
            List of parameters not contained in this optimizer param groups
        """
        out_params = []
        removed_params = []
        for param in params:
            if not any(torch.equal(p, param) for group in optimizer.param_groups for p in group["params"]):
                out_params.append(param)
            else:
                removed_params.append(param)

        if removed_params:
            rank_zero_warn(
                "The provided params to be frozen already exist within another group of this optimizer."
                " Those parameters will be skipped.\n"
                "HINT: Did you init your optimizer in `configure_optimizer` as such:\n"
                f" {type(optimizer)}(filter(lambda p: p.requires_grad, self.parameters()), ...) ",
            )
        return out_params

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.
        """
        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({"params": params, "lr": params_lr / denom_lr})

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.freeze_before_training(pl_module)

    @staticmethod
    def _apply_mapping_to_param_groups(param_groups: List[Dict[str, Any]], mapping: dict) -> List[Dict[str, Any]]:
        output = []
        for g in param_groups:
            # skip params to save memory
            group_state = {k: v for k, v in g.items() if k != "params"}
            group_state["params"] = [mapping[p] for p in g["params"]]
            output.append(group_state)
        return output

    def _store(
        self,
        pl_module: "pl.LightningModule",
        opt_idx: int,
        num_param_groups: int,
        current_param_groups: List[Dict[str, Any]],
    ) -> None:
        mapping = {p: n for n, p in pl_module.named_parameters()}
        if opt_idx not in self._internal_optimizer_metadata:
            self._internal_optimizer_metadata[opt_idx] = self._apply_mapping_to_param_groups(
                current_param_groups, mapping
            )
        elif num_param_groups != len(current_param_groups):
            # save new param_groups possibly created by the users.
            self._internal_optimizer_metadata[opt_idx].extend(
                self._apply_mapping_to_param_groups(current_param_groups[num_param_groups:], mapping)
            )

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the epoch begins."""
        # import is here to avoid circular imports
        from pytorch_lightning.loops.utilities import _get_active_optimizers

        for opt_idx, optimizer in _get_active_optimizers(trainer.optimizers, trainer.optimizer_frequencies, 0):
            num_param_groups = len(optimizer.param_groups)
            self.finetune_function(pl_module, trainer.current_epoch, optimizer, opt_idx)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Override to add your unfreeze logic."""
        raise NotImplementedError

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        """Override to add your freeze logic."""
        raise NotImplementedError


class BackboneFinetuning(BaseFinetuning):
    r"""Finetune a backbone model based on a learning rate user-defined scheduling.

    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Initial learning rate for the backbone.
            By default, we will use ``current_learning /  backbone_initial_ratio_lr``
        should_align: Whether to align with current learning rate when backbone learning
            reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial learning rate will
            ``current_learning_rate /  initial_denom_lr``.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone
        rounding: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        super().load_state_dict(state_dict)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.backbone)

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )
