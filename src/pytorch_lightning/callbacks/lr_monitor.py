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

Learning Rate Monitor
=====================

Monitor and logs learning rate for lr schedulers during training.

"""
import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Type

from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig


class LearningRateMonitor(Callback):
    r"""
    Automatically monitor and logs learning rate for learning rate schedulers during training.

    Args:
        logging_interval: set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers
            at the same interval, set to ``None`` to log at individual interval
            according to the ``interval`` key of each scheduler. Defaults to ``None``.
        log_momentum: option to also log the momentum values of the optimizer, if the optimizer
            has the ``momentum`` or ``betas`` attribute. Defaults to ``False``.

    Raises:
        MisconfigurationException:
            If ``logging_interval`` is none of ``"step"``, ``"epoch"``, or ``None``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LearningRateMonitor
        >>> lr_monitor = LearningRateMonitor(logging_interval='step')
        >>> trainer = Trainer(callbacks=[lr_monitor])

    Logging names are automatically determined based on optimizer class name.
    In case of multiple optimizers of same type, they will be named ``Adam``,
    ``Adam-1`` etc. If a optimizer has multiple parameter groups they will
    be named ``Adam/pg1``, ``Adam/pg2`` etc. To control naming, pass in a
    ``name`` keyword in the construction of the learning rate schedulers.
    A ``name`` keyword can also be used for parameter groups in the
    construction of the optimizer.

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.Adam(...)
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
                'name': 'my_logging_name'
            }
            return [optimizer], [lr_scheduler]

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.SGD(
                [{
                    'params': [p for p in self.parameters()],
                    'name': 'my_parameter_group_name'
                }],
                lr=0.1
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
            return [optimizer], [lr_scheduler]

    """

    def __init__(self, logging_interval: Optional[str] = None, log_momentum: bool = False) -> None:
        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException("logging_interval should be `step` or `epoch` or `None`.")

        self.logging_interval = logging_interval
        self.log_momentum = log_momentum
        self.lrs: Dict[str, List[float]] = {}

    def on_train_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        """Called before training, determines unique names for all lr schedulers in the case of multiple of the
        same type or in the case of multiple parameter groups.

        Raises:
            MisconfigurationException:
                If ``Trainer`` has no ``logger``.
        """
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger."
            )

        if self.log_momentum:

            def _check_no_key(key: str) -> bool:
                if trainer.lr_scheduler_configs:
                    return any(
                        key not in config.scheduler.optimizer.defaults for config in trainer.lr_scheduler_configs
                    )

                return any(key not in optimizer.defaults for optimizer in trainer.optimizers)

            if _check_no_key("momentum") and _check_no_key("betas"):
                rank_zero_warn(
                    "You have set log_momentum=True, but some optimizers do not"
                    " have momentum. This will log a value 0 for the momentum.",
                    category=RuntimeWarning,
                )

        # Find names for schedulers
        names: List[List[str]] = []
        (
            sched_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(trainer.lr_scheduler_configs)
        names.extend(sched_hparam_keys)

        # Find names for leftover optimizers
        optimizer_hparam_keys, _ = self._find_names_from_optimizers(
            trainer.optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        names.extend(optimizer_hparam_keys)

        # Initialize for storing values
        names_flatten = list(itertools.chain.from_iterable(names))
        self.lrs = {name: [] for name in names_flatten}
        self.last_momentum_values = {name + "-momentum": None for name in names_flatten}

    def on_train_batch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        if self.logging_interval != "epoch":
            interval = "step" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                for logger in trainer.loggers:
                    logger.log_metrics(latest_stat, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                for logger in trainer.loggers:
                    logger.log_metrics(latest_stat, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def _extract_stats(self, trainer: "pl.Trainer", interval: str) -> Dict[str, float]:
        latest_stat = {}

        (
            scheduler_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(trainer.lr_scheduler_configs)
        self._remap_keys(scheduler_hparam_keys)

        for name, config in zip(scheduler_hparam_keys, trainer.lr_scheduler_configs):
            if interval in [config.interval, "any"]:
                opt = config.scheduler.optimizer
                current_stat = self._get_lr_momentum_stat(opt, name)
                latest_stat.update(current_stat)

        optimizer_hparam_keys, optimizers_without_scheduler = self._find_names_from_optimizers(
            trainer.optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        self._remap_keys(optimizer_hparam_keys)

        for opt, names in zip(optimizers_without_scheduler, optimizer_hparam_keys):
            current_stat = self._get_lr_momentum_stat(opt, names)
            latest_stat.update(current_stat)

        return latest_stat

    def _get_lr_momentum_stat(self, optimizer: Optimizer, names: List[str]) -> Dict[str, float]:
        lr_momentum_stat = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults

        for pg, name in zip(param_groups, names):
            lr = self._extract_lr(pg, name)
            lr_momentum_stat.update(lr)
            momentum = self._extract_momentum(
                param_group=pg, name=name.replace(name, f"{name}-momentum"), use_betas=use_betas
            )
            lr_momentum_stat.update(momentum)

        return lr_momentum_stat

    def _extract_lr(self, param_group: Dict[str, Any], name: str) -> Dict[str, Any]:
        lr = param_group["lr"]
        self.lrs[name].append(lr)
        return {name: lr}

    def _remap_keys(self, names: List[List[str]], token: str = "/pg1") -> None:
        """This function is used the remap the keys if param groups for a given optimizer increased."""
        for group_new_names in names:
            for new_name in group_new_names:
                old_name = new_name.replace(token, "")
                if token in new_name and old_name in self.lrs:
                    self.lrs[new_name] = self.lrs.pop(old_name)
                elif new_name not in self.lrs:
                    self.lrs[new_name] = []

    def _extract_momentum(self, param_group: Dict[str, List], name: str, use_betas: bool) -> Dict[str, float]:
        if not self.log_momentum:
            return {}

        momentum = param_group["betas"][0] if use_betas else param_group.get("momentum", 0)
        self.last_momentum_values[name] = momentum
        return {name: momentum}

    def _add_prefix(
        self, name: str, optimizer_cls: Type[Optimizer], seen_optimizer_types: DefaultDict[Type[Optimizer], int]
    ) -> str:
        if optimizer_cls not in seen_optimizer_types:
            return name
        count = seen_optimizer_types[optimizer_cls]
        return name + f"-{count - 1}" if count > 1 else name

    def _add_suffix(self, name: str, param_groups: List[Dict], param_group_index: int, use_names: bool = True) -> str:
        if len(param_groups) > 1:
            if not use_names:
                return f"{name}/pg{param_group_index+1}"
            pg_name = param_groups[param_group_index].get("name", f"pg{param_group_index+1}")
            return f"{name}/{pg_name}"
        elif use_names:
            pg_name = param_groups[param_group_index].get("name")
            return f"{name}/{pg_name}" if pg_name else name
        return name

    def _duplicate_param_group_names(self, param_groups: List[Dict]) -> Set[str]:
        names = [pg.get("name", f"pg{i}") for i, pg in enumerate(param_groups, start=1)]
        unique = set(names)
        if len(names) == len(unique):
            return set()
        return {n for n in names if names.count(n) > 1}

    def _find_names_from_schedulers(
        self,
        lr_scheduler_configs: List[LRSchedulerConfig],
    ) -> Tuple[List[List[str]], List[Optimizer], DefaultDict[Type[Optimizer], int]]:
        # Create unique names in the case we have multiple of the same learning
        # rate scheduler + multiple parameter groups
        names = []
        seen_optimizers: List[Optimizer] = []
        seen_optimizer_types: DefaultDict[Type[Optimizer], int] = defaultdict(int)
        for config in lr_scheduler_configs:
            sch = config.scheduler
            if config.name is not None:
                name = config.name
            else:
                name = "lr-" + sch.optimizer.__class__.__name__

            updated_names = self._check_duplicates_and_update_name(
                sch.optimizer, name, seen_optimizers, seen_optimizer_types, config
            )
            names.append(updated_names)

        return names, seen_optimizers, seen_optimizer_types

    def _find_names_from_optimizers(
        self,
        optimizers: List[Any],
        seen_optimizers: List[Optimizer],
        seen_optimizer_types: DefaultDict[Type[Optimizer], int],
    ) -> Tuple[List[List[str]], List[Optimizer]]:
        names = []
        optimizers_without_scheduler = []

        for optimizer in optimizers:
            # Deepspeed optimizer wraps the native optimizer
            optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            if optimizer in seen_optimizers:
                continue

            name = "lr-" + optimizer.__class__.__name__
            updated_names = self._check_duplicates_and_update_name(
                optimizer, name, seen_optimizers, seen_optimizer_types, None
            )
            names.append(updated_names)
            optimizers_without_scheduler.append(optimizer)

        return names, optimizers_without_scheduler

    def _check_duplicates_and_update_name(
        self,
        optimizer: Optimizer,
        name: str,
        seen_optimizers: List[Optimizer],
        seen_optimizer_types: DefaultDict[Type[Optimizer], int],
        lr_scheduler_config: Optional[LRSchedulerConfig],
    ) -> List[str]:
        seen_optimizers.append(optimizer)
        optimizer_cls = type(optimizer)
        if lr_scheduler_config is not None and lr_scheduler_config.name is None:
            seen_optimizer_types[optimizer_cls] += 1
        elif lr_scheduler_config is None:
            seen_optimizer_types[optimizer_cls] += 1

        # Multiple param groups for the same optimizer
        param_groups = optimizer.param_groups
        duplicates = self._duplicate_param_group_names(param_groups)
        if duplicates:
            raise MisconfigurationException(
                "A single `Optimizer` cannot have multiple parameter groups with identical "
                f"`name` values. {name} has duplicated parameter group names {duplicates}"
            )

        name = self._add_prefix(name, optimizer_cls, seen_optimizer_types)
        name_list = [self._add_suffix(name, param_groups, i) for i in range(len(param_groups))]

        return name_list
