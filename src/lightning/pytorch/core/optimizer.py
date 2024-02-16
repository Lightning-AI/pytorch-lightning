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
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy

import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature
from lightning.pytorch.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple


def do_nothing_closure() -> None:
    return


class LightningOptimizer:
    """This class is used to wrap the user optimizers and handle properly the backward and optimizer_step logic across
    accelerators, AMP, accumulate_grad_batches.

    Note: The purpose of this wrapper is only to define new methods and redirect the `.step()` call. The internal
    state ``__dict__`` is not kept in sync with the internal state of the original optimizer, but the Trainer never
    relies on the internal state of the wrapper.

    """

    def __init__(self, optimizer: Optimizer):
        self._optimizer = optimizer
        self._strategy: Optional[pl.strategies.Strategy] = None
        # to inject logic around the optimizer step, particularly useful with manual optimization
        self._on_before_step = do_nothing_closure
        self._on_after_step = do_nothing_closure
        # imitate the class of the wrapped object to make isinstance checks work
        self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @contextmanager
    def toggle_model(self, sync_grad: bool = True) -> Generator[None, None, None]:
        """This function is just a helper for advanced users.

        Considering the current optimizer as A and all other optimizers as B.
        Toggling means all parameters from B exclusive to A will have ``requires_grad`` set to False.

        When performing gradient accumulation, there is no need to perform grad synchronization
        during the accumulation phase.
        Setting `sync_grad` to False will block this synchronization and improve performance.

        """
        # local import here to avoid circular import
        from lightning.pytorch.loops.utilities import _block_parallel_sync_behavior

        assert self._strategy is not None
        lightning_module = self._strategy.lightning_module
        assert lightning_module is not None
        with _block_parallel_sync_behavior(self._strategy, block=(not sync_grad)):
            lightning_module.toggle_optimizer(self)
            yield
            lightning_module.untoggle_optimizer(self)

    def step(self, closure: Optional[Callable[[], Any]] = None, **kwargs: Any) -> Any:
        """Performs a single optimization step (parameter update).

        Args:
            closure: An optional optimizer closure.
            kwargs: Any additional arguments to the ``optimizer.step()`` call.

        Returns:
            The output from the step call, which is generally the output of the closure execution.

        Example::

            # Scenario for a GAN using manual optimization
            def training_step(self, batch, batch_idx):
                opt_gen, opt_dis = self.optimizers()

                ...

                # compute generator loss
                loss_gen = self.compute_generator_loss(...)
                # zero_grad needs to be called before backward
                opt_gen.zero_grad()
                self.manual_backward(loss_gen)
                opt_gen.step()

                # compute discriminator loss
                loss_dis = self.compute_discriminator_loss(...)

                # zero_grad needs to be called before backward
                opt_dis.zero_grad()
                self.manual_backward(loss_dis)
                opt_dis.step()


            # A more advanced example
            def training_step(self, batch, batch_idx):
                opt_gen, opt_dis = self.optimizers()

                ...
                accumulated_grad_batches = batch_idx % 2 == 0

                # compute generator loss
                def closure_gen():
                    loss_gen = self.compute_generator_loss(...)
                    self.manual_backward(loss_gen)
                    if accumulated_grad_batches:
                        opt_gen.zero_grad()

                with opt_gen.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_gen.step(closure=closure_gen)

                def closure_dis():
                    loss_dis = self.compute_discriminator_loss(...)
                    self.manual_backward(loss_dis)
                    if accumulated_grad_batches:
                        opt_dis.zero_grad()

                with opt_dis.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_dis.step(closure=closure_dis)

        """
        self._on_before_step()

        if closure is None:
            closure = do_nothing_closure
        elif not callable(closure):
            raise MisconfigurationException("When `optimizer.step(closure)` is called, the closure should be callable")

        assert self._strategy is not None
        step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)

        self._on_after_step()

        return step_output

    @classmethod
    def _to_lightning_optimizer(
        cls, optimizer: Union[Optimizer, "LightningOptimizer"], strategy: "pl.strategies.Strategy"
    ) -> "LightningOptimizer":
        # the user could return a `LightningOptimizer` from `configure_optimizers`, see test:
        # tests/core/test_lightning_optimizer.py::test_lightning_optimizer[False]
        lightning_optimizer = optimizer if isinstance(optimizer, LightningOptimizer) else cls(optimizer)
        lightning_optimizer._strategy = proxy(strategy)
        return lightning_optimizer

    def __getattr__(self, item: Any) -> Any:
        return getattr(self._optimizer, item)


def _init_optimizers_and_lr_schedulers(
    model: "pl.LightningModule",
) -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
    """Calls `LightningModule.configure_optimizers` and parses and validates the output."""
    from lightning.pytorch.trainer import call

    optim_conf = call._call_lightning_module_hook(model.trainer, "configure_optimizers", pl_module=model)

    if optim_conf is None:
        rank_zero_warn(
            "`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer",
        )
        optim_conf = _MockOptimizer()

    optimizers, lr_schedulers, monitor = _configure_optimizers(optim_conf)
    lr_scheduler_configs = (
        _configure_schedulers_automatic_opt(lr_schedulers, monitor)
        if model.automatic_optimization
        else _configure_schedulers_manual_opt(lr_schedulers)
    )
    _validate_multiple_optimizers_support(optimizers, model)
    _validate_optimizers_attached(optimizers, lr_scheduler_configs)
    _validate_scheduler_api(lr_scheduler_configs, model)
    return optimizers, lr_scheduler_configs


def _configure_optimizers(
    optim_conf: Union[Dict[str, Any], List, Optimizer, Tuple],
) -> Tuple[List, List, Optional[str]]:
    optimizers, lr_schedulers = [], []
    monitor = None

    # single output, single optimizer
    if isinstance(optim_conf, Optimizable):
        optimizers = [optim_conf]
    # two lists, optimizer + lr schedulers
    elif (
        isinstance(optim_conf, (list, tuple))
        and len(optim_conf) == 2
        and isinstance(optim_conf[0], list)
        and all(isinstance(opt, Optimizable) for opt in optim_conf[0])
    ):
        opt, sch = optim_conf
        optimizers = opt
        lr_schedulers = sch if isinstance(sch, list) else [sch]
    # single dictionary
    elif isinstance(optim_conf, dict):
        _validate_optim_conf(optim_conf)
        optimizers = [optim_conf["optimizer"]]
        monitor = optim_conf.get("monitor", None)
        lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
    # multiple dictionaries
    elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
        for opt_dict in optim_conf:
            _validate_optim_conf(opt_dict)
        optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
        scheduler_dict = lambda scheduler: dict(scheduler) if isinstance(scheduler, dict) else {"scheduler": scheduler}
        lr_schedulers = [
            scheduler_dict(opt_dict["lr_scheduler"]) for opt_dict in optim_conf if "lr_scheduler" in opt_dict
        ]
    # single list or tuple, multiple optimizer
    elif isinstance(optim_conf, (list, tuple)) and all(isinstance(opt, Optimizable) for opt in optim_conf):
        optimizers = list(optim_conf)
    # unknown configuration
    else:
        raise MisconfigurationException(
            "Unknown configuration for model optimizers."
            " Output from `model.configure_optimizers()` should be one of:\n"
            " * `Optimizer`\n"
            " * [`Optimizer`]\n"
            " * ([`Optimizer`], [`LRScheduler`])\n"
            ' * {"optimizer": `Optimizer`, (optional) "lr_scheduler": `LRScheduler`}\n'
        )
    return optimizers, lr_schedulers, monitor


def _configure_schedulers_automatic_opt(schedulers: list, monitor: Optional[str]) -> List[LRSchedulerConfig]:
    """Convert each scheduler into `LRSchedulerConfig` with relevant information, when using automatic optimization."""
    lr_scheduler_configs = []
    for scheduler in schedulers:
        if isinstance(scheduler, dict):
            # check provided keys
            supported_keys = {field.name for field in fields(LRSchedulerConfig)}
            extra_keys = scheduler.keys() - supported_keys
            if extra_keys:
                rank_zero_warn(
                    f"Found unsupported keys in the lr scheduler dict: {extra_keys}."
                    " HINT: remove them from the output of `configure_optimizers`.",
                    category=RuntimeWarning,
                )
                scheduler = {k: v for k, v in scheduler.items() if k in supported_keys}
            if "scheduler" not in scheduler:
                raise MisconfigurationException(
                    'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                )
            if "interval" in scheduler and scheduler["interval"] not in ("step", "epoch"):
                raise MisconfigurationException(
                    'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                    f' but is "{scheduler["interval"]}"'
                )
            scheduler["reduce_on_plateau"] = scheduler.get(
                "reduce_on_plateau", isinstance(scheduler["scheduler"], optim.lr_scheduler.ReduceLROnPlateau)
            )
            if scheduler["reduce_on_plateau"] and scheduler.get("monitor", None) is None:
                raise MisconfigurationException(
                    "The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used."
                    ' For example: {"optimizer": optimizer, "lr_scheduler":'
                    ' {"scheduler": scheduler, "monitor": "your_loss"}}'
                )
            is_one_cycle = isinstance(scheduler["scheduler"], optim.lr_scheduler.OneCycleLR)
            if is_one_cycle and scheduler.get("interval", "epoch") == "epoch":
                rank_zero_warn(
                    "A `OneCycleLR` scheduler is using 'interval': 'epoch'."
                    " Are you sure you didn't mean 'interval': 'step'?",
                    category=RuntimeWarning,
                )
            config = LRSchedulerConfig(**scheduler)
        elif isinstance(scheduler, ReduceLROnPlateau):
            if monitor is None:
                raise MisconfigurationException(
                    "`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
                    " scheduler is used. For example:"
                    ' {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                )
            config = LRSchedulerConfig(scheduler, reduce_on_plateau=True, monitor=monitor)
        else:
            config = LRSchedulerConfig(scheduler)
        lr_scheduler_configs.append(config)
    return lr_scheduler_configs


def _configure_schedulers_manual_opt(schedulers: list) -> List[LRSchedulerConfig]:
    """Convert each scheduler into `LRSchedulerConfig` structure with relevant information, when using manual
    optimization."""
    lr_scheduler_configs = []
    for scheduler in schedulers:
        if isinstance(scheduler, dict):
            # interval is not in this list even though the user needs to manually call the scheduler because
            # the `LearningRateMonitor` callback needs to check its value to know when to log the learning rate
            invalid_keys = {"reduce_on_plateau", "monitor", "strict"}
            keys_to_warn = [k for k in scheduler if k in invalid_keys]

            if keys_to_warn:
                rank_zero_warn(
                    f"The lr scheduler dict contains the key(s) {keys_to_warn}, but the keys will be ignored."
                    " You need to call `lr_scheduler.step()` manually in manual optimization.",
                    category=RuntimeWarning,
                )

            config = LRSchedulerConfig(**{key: scheduler[key] for key in scheduler if key not in invalid_keys})
        else:
            config = LRSchedulerConfig(scheduler)
        lr_scheduler_configs.append(config)
    return lr_scheduler_configs


def _validate_scheduler_api(lr_scheduler_configs: List[LRSchedulerConfig], model: "pl.LightningModule") -> None:
    for config in lr_scheduler_configs:
        scheduler = config.scheduler
        if not isinstance(scheduler, _Stateful):
            raise TypeError(
                f"The provided lr scheduler `{scheduler.__class__.__name__}` is invalid."
                " It should have `state_dict` and `load_state_dict` methods defined."
            )

        if (
            not isinstance(scheduler, LRSchedulerTypeTuple)
            and not is_overridden("lr_scheduler_step", model)
            and model.automatic_optimization
        ):
            raise MisconfigurationException(
                f"The provided lr scheduler `{scheduler.__class__.__name__}` doesn't follow PyTorch's LRScheduler"
                " API. You should override the `LightningModule.lr_scheduler_step` hook with your own logic if"
                " you are using a custom LR scheduler."
            )


def _validate_multiple_optimizers_support(optimizers: List[Optimizer], model: "pl.LightningModule") -> None:
    if is_param_in_hook_signature(model.training_step, "optimizer_idx", explicit=True):
        raise RuntimeError(
            "Training with multiple optimizers is only supported with manual optimization. Remove the `optimizer_idx`"
            " argument from `training_step`, set `self.automatic_optimization = False` and access your optimizers"
            " in `training_step` with `opt1, opt2, ... = self.optimizers()`."
        )
    if model.automatic_optimization and len(optimizers) > 1:
        raise RuntimeError(
            "Training with multiple optimizers is only supported with manual optimization. Set"
            " `self.automatic_optimization = False`, then access your optimizers in `training_step` with"
            " `opt1, opt2, ... = self.optimizers()`."
        )


def _validate_optimizers_attached(optimizers: List[Optimizer], lr_scheduler_configs: List[LRSchedulerConfig]) -> None:
    for config in lr_scheduler_configs:
        if config.scheduler.optimizer not in optimizers:
            raise MisconfigurationException(
                "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
            )


def _validate_optim_conf(optim_conf: Dict[str, Any]) -> None:
    valid_keys = {"optimizer", "lr_scheduler", "monitor"}
    extra_keys = optim_conf.keys() - valid_keys
    if extra_keys:
        rank_zero_warn(
            f"Found unsupported keys in the optimizer configuration: {set(extra_keys)}", category=RuntimeWarning
        )


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None` is returned from
    :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`."""

    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    @override
    def add_param_group(self, param_group: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    @override
    def load_state_dict(self, state_dict: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    @override
    def state_dict(self) -> Dict[str, Any]:
        return {}  # Return Empty

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            return closure()

    @override
    def zero_grad(self, set_to_none: Optional[bool] = True) -> None:
        pass  # Do Nothing

    @override
    def __repr__(self) -> str:
        return "No Optimizer"
