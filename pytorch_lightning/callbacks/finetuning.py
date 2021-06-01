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
Finetuning Callback
^^^^^^^^^^^^^^^^^^^^
Freeze and unfreeze models for finetuning purposes
"""
import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


def multiplicative(epoch):
    return 2


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

        class MyModel(LightningModule)

            ...

            def configure_optimizer(self):
                # Make sure to filter the parameters based on `requires_grad`
                return Adam(filter(lambda p: p.requires_grad, self.parameters))

        class FeatureExtractorFreezeUnfreeze(BaseFinetuning):

            def __init__(self, unfreeze_at_epoch=10)
                self._unfreeze_at_epoch = unfreeze_at_epoch

            def freeze_before_training(self, pl_module):
                # freeze any module you want
                # Here, we are freezing ``feature_extractor``
                self.freeze(pl_module.feature_extractor)

            def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
                # When `current_epoch` is 10, feature_extractor will start training.
                if current_epoch == self._unfreeze_at_epoch:
                    self.unfreeze_and_add_param_group(
                        modules=pl_module.feature_extractor,
                        optimizer=optimizer,
                        train_bn=True,
                    )
    """

    def __init__(self):
        self._internal_state: Dict[int, List[Dict[str, Any]]] = {}

    def on_save_checkpoint(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        checkpoint: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        return self._internal_state

    def on_load_checkpoint(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', callback_state: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        self._internal_state = callback_state
        # restore the param_groups created during the previous training.
        named_parameters = dict(pl_module.named_parameters())
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            param_groups = self.__apply_mapping_to_param_groups(self._internal_state[opt_idx], named_parameters)
            optimizer.param_groups = param_groups

    @staticmethod
    def flatten_modules(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> List[Module]:
        """
        This function is used to flatten a module or an iterable of modules into a list of its modules.

        Args:
            modules: A given module or an iterable of modules

        Returns:
            List of modules
        """
        if isinstance(modules, Iterable):
            _modules = []
            for m in modules:
                _modules.extend(BaseFinetuning.flatten_modules(m))

        else:
            _modules = modules.modules()

        # Leaf nodes in the graph have no children, so we use that to filter
        return [m for m in _modules if not list(m.children())]

    @staticmethod
    def filter_params(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        train_bn: bool = True,
        requires_grad: bool = True
    ) -> Generator:
        """Yields the `requires_grad` parameters of a given module or list of modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: Whether to train BatchNorm module
            requires_grad: Whether to create a generator for trainable or non-trainable parameters.

        Returns:
            Generator
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and not train_bn:
                continue
            for param in mod.parameters():
                if param.requires_grad == requires_grad:
                    yield param

    @staticmethod
    def make_trainable(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        """
        Unfreezes the parameters of the provided modules

        Args:
            modules: A given module or an iterable of modules
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    @staticmethod
    def freeze(modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True) -> None:
        """
        Freezes the parameters of the provided modules

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
                for param in mod.parameters():
                    param.requires_grad = False

    @staticmethod
    def filter_on_optimizer(optimizer: Optimizer, params: Iterable) -> List:
        """
        This function is used to exclude any parameter which already exists in
        this optimizer

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
                "The provided params to be freezed already exist within another group of this optimizer."
                " Those parameters will be skipped.\n"
                "HINT: Did you init your optimizer in `configure_optimizer` as such:\n"
                f" {type(optimizer)}(filter(lambda p: p.requires_grad, self.parameters()), ...) ", UserWarning
            )
        return out_params

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.,
        train_bn: bool = True,
    ) -> None:
        """
        Unfreezes a module and adds its parameters to an optimizer.

        Args:

            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.

            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`

            lr: Learning rate for the new param group.

            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by initial_denom_lr.

            train_bn: Whether to train the BatchNormalization layers.

        Returns:
            None
        """
        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({
                'params': params,
                'lr': params_lr / denom_lr,
            })

    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        self.freeze_before_training(pl_module)

    @staticmethod
    def __apply_mapping_to_param_groups(param_groups: List[Dict[str, Any]], mapping: dict) -> List[Dict[str, Any]]:
        output = []
        for g in param_groups:
            # skip params to save memory
            group_state = {k: v for k, v in g.items() if k != 'params'}
            group_state['params'] = [mapping[p] for p in g['params']]
            output.append(group_state)
        return output

    def _store(
        self,
        pl_module: 'pl.LightningModule',
        opt_idx: int,
        num_param_groups: int,
        current_param_groups: List[Dict[str, Any]],
    ) -> None:
        mapping = {p: n for n, p in pl_module.named_parameters()}
        if opt_idx not in self._internal_state:
            self._internal_state[opt_idx] = self.__apply_mapping_to_param_groups(current_param_groups, mapping)
        elif num_param_groups != len(current_param_groups):
            # save new param_groups possibly created by the users.
            self._internal_state[opt_idx].extend(
                self.__apply_mapping_to_param_groups(current_param_groups[num_param_groups:], mapping)
            )

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        for opt_idx, optimizer in trainer.train_loop.get_active_optimizers():
            num_param_groups = len(optimizer.param_groups)
            self.finetune_function(pl_module, trainer.current_epoch, optimizer, opt_idx)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def finetune_function(self, pl_module: 'pl.LightningModule', epoch: int, optimizer: Optimizer, opt_idx: int):
        """
        Override to add your unfreeze logic
        """
        raise NotImplementedError

    def freeze_before_training(self, pl_module: 'pl.LightningModule'):
        """
        Override to add your freeze logic
        """
        raise NotImplementedError


class BackboneFinetuning(BaseFinetuning):
    r"""

    Finetune a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for model and backbone

        round: Precision for displaying learning rate

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
        initial_denom_lr: float = 10.,
        train_bn: bool = True,
        verbose: bool = False,
        round: int = 12,
    ):
        super().__init__()

        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.backbone_initial_lr = backbone_initial_lr
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.round = round
        self.verbose = verbose

    def on_fit_start(self, trainer, pl_module):
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: 'pl.LightningModule'):
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module: 'pl.LightningModule', epoch: int, optimizer: Optimizer, opt_idx: int):
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]['lr']
            initial_backbone_lr = self.backbone_initial_lr if self.backbone_initial_lr is not None \
                else current_lr * self.backbone_initial_ratio_lr
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.round)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]['lr']
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = current_lr if (self.should_align and next_current_backbone_lr > current_lr) \
                else next_current_backbone_lr
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.round)}"
                )
