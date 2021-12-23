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
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from weakref import proxy

import torch
from torch import optim
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def do_nothing_closure() -> None:
    return


class LightningOptimizer:
    """This class is used to wrap the user optimizers and handle properly the backward and optimizer_step logic
    across accelerators, AMP, accumulate_grad_batches."""

    def __init__(self, optimizer: Optimizer):
        # copy most of the `Optimizer` methods into this instance. `__del__` is skipped in case the optimizer has
        # implemented custom logic which we would not want to call on destruction of the `LightningOptimizer`
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}

        # For Horovod
        if hasattr(optimizer, "skip_synchronize"):
            self.__class__ = type(
                "Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__.__bases__[0]), {}
            )
            self.skip_synchronize = optimizer.skip_synchronize
            self.synchronize = optimizer.synchronize
        else:
            self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

        self._optimizer = optimizer
        self._trainer: Optional["pl.Trainer"] = None
        self._optimizer_idx = 0

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def _on_trainer_init(self, trainer: "pl.Trainer") -> None:
        # check if trainer is already of type weakproxy since we can't call proxy on a weakproxy
        self._trainer = trainer if isinstance(trainer, weakref.ProxyType) else proxy(trainer)
        for opt_idx, opt in enumerate(trainer.optimizers):
            if opt == self._optimizer:
                self._optimizer_idx = opt_idx
                break

    @classmethod
    def _to_lightning_optimizer(cls, optimizer: Optimizer, trainer: "pl.Trainer", opt_idx: int) -> "LightningOptimizer":
        # apex overrides .step function and need to be wrapped on each step
        if trainer.amp_backend is not None and trainer.amp_backend == AMPType.APEX:
            lightning_optimizer = cls(optimizer)
            lightning_optimizer._on_trainer_init(trainer)
        else:
            lightning_optimizer = trainer.lightning_optimizers[opt_idx]
        return lightning_optimizer

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
        from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior

        assert self._trainer is not None
        lightning_module = self._trainer.lightning_module

        with _block_parallel_sync_behavior(self._trainer, block=(not sync_grad)):
            lightning_module.toggle_optimizer(self, self._optimizer_idx)
            yield
            lightning_module.untoggle_optimizer(self._optimizer_idx)

    def step(self, closure: Optional[Callable[[], Any]] = None, **kwargs: Any) -> None:
        """Performs a single optimization step (parameter update).

        Args:
            closure: An optional optimizer_closure.
            kwargs: Any additional arguments to the ``optimizer.step()`` call.

        Example::

            # Scenario for a GAN using manual optimization
            def training_step(...):
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
            def training_step(self, batch, batch_idx, ...):
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
        if closure is None:
            closure = do_nothing_closure
            profiler_action = "optimizer_step_without_closure"
        elif not callable(closure):
            raise MisconfigurationException("When `optimizer.step(closure)` is called, the closure should be callable")
        else:
            profiler_action = "optimizer_step_with_closure"
        profiler_action += f"_{self._optimizer_idx}"

        trainer = self._trainer
        assert trainer is not None
        with trainer.profiler.profile(profiler_action):
            trainer.strategy.optimizer_step(self._optimizer, self._optimizer_idx, closure, **kwargs)


def _init_optimizers_and_lr_schedulers(model: "pl.LightningModule") -> Tuple[List, List, List]:
    """Calls `LightningModule.configure_optimizers` and parses and validates the output."""
    model.trainer._lightning_optimizers = None
    optim_conf = model.trainer._call_lightning_module_hook("configure_optimizers", pl_module=model)

    if optim_conf is None:
        rank_zero_warn(
            "`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer",
        )
        optim_conf = _MockOptimizer()

    optimizers, lr_schedulers, optimizer_frequencies, monitor = _configure_optimizers(optim_conf)
    lr_schedulers = _configure_schedulers(lr_schedulers, monitor, not model.automatic_optimization)
    _validate_scheduler_optimizer(optimizers, lr_schedulers)
    return optimizers, lr_schedulers, optimizer_frequencies


def _configure_optimizers(
    optim_conf: Union[Dict[str, Any], List, Optimizer, Tuple]
) -> Tuple[List, List, List, Optional[str]]:
    optimizers, lr_schedulers, optimizer_frequencies = [], [], []
    monitor = None

    # single output, single optimizer
    if isinstance(optim_conf, Optimizer):
        optimizers = [optim_conf]
    # two lists, optimizer + lr schedulers
    elif (
        isinstance(optim_conf, (list, tuple))
        and len(optim_conf) == 2
        and isinstance(optim_conf[0], list)
        and all(isinstance(opt, Optimizer) for opt in optim_conf[0])
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
        scheduler_dict = (
            lambda scheduler, opt_idx: dict(scheduler, opt_idx=opt_idx)
            if isinstance(scheduler, dict)
            else {"scheduler": scheduler, "opt_idx": opt_idx}
        )

        lr_schedulers = [
            scheduler_dict(opt_dict["lr_scheduler"], opt_idx)
            for opt_idx, opt_dict in enumerate(optim_conf)
            if "lr_scheduler" in opt_dict
        ]
        optimizer_frequencies = [
            opt_dict["frequency"] for opt_dict in optim_conf if opt_dict.get("frequency", None) is not None
        ]
        # assert that if frequencies are present, they are given for all optimizers
        if optimizer_frequencies and len(optimizer_frequencies) != len(optimizers):
            raise ValueError("A frequency must be given to each optimizer.")
    # single list or tuple, multiple optimizer
    elif isinstance(optim_conf, (list, tuple)) and all(isinstance(opt, Optimizer) for opt in optim_conf):
        optimizers = list(optim_conf)
    # unknown configuration
    else:
        raise MisconfigurationException(
            "Unknown configuration for model optimizers."
            " Output from `model.configure_optimizers()` should be one of:\n"
            " * `Optimizer`\n"
            " * [`Optimizer`]\n"
            " * ([`Optimizer`], [`_LRScheduler`])\n"
            ' * {"optimizer": `Optimizer`, (optional) "lr_scheduler": `_LRScheduler`}\n'
            ' * A list of the previously described dict format, with an optional "frequency" key (int)'
        )
    return optimizers, lr_schedulers, optimizer_frequencies, monitor


def _configure_schedulers(
    schedulers: list, monitor: Optional[str], is_manual_optimization: bool
) -> List[Dict[str, Any]]:
    """Convert each scheduler into dict structure with relevant information."""
    lr_schedulers = []
    default_config = _get_default_scheduler_config()
    # TODO: move is_manual_optimization check out of for loop
    for scheduler in schedulers:
        if is_manual_optimization:
            if isinstance(scheduler, dict):
                invalid_keys = {"interval", "frequency", "reduce_on_plateau", "monitor", "strict"}
                keys_to_warn = [k for k in scheduler.keys() if k in invalid_keys]

                if keys_to_warn:
                    rank_zero_warn(
                        f"The lr scheduler dict contains the key(s) {keys_to_warn}, but the keys will be ignored."
                        " You need to call `lr_scheduler.step()` manually in manual optimization.",
                        category=RuntimeWarning,
                    )

                scheduler = {key: scheduler[key] for key in scheduler if key not in invalid_keys}
                lr_schedulers.append({**default_config, **scheduler})
            else:
                lr_schedulers.append({**default_config, "scheduler": scheduler})
        else:
            if isinstance(scheduler, dict):
                # check provided keys
                extra_keys = [k for k in scheduler.keys() if k not in default_config.keys()]
                if extra_keys:
                    rank_zero_warn(
                        f"Found unsupported keys in the lr scheduler dict: {extra_keys}", category=RuntimeWarning
                    )
                if "scheduler" not in scheduler:
                    raise MisconfigurationException(
                        'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                    )
                if "interval" in scheduler and scheduler["interval"] not in ("step", "epoch"):
                    raise MisconfigurationException(
                        'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                        f' but is "{scheduler["interval"]}"'
                    )
                scheduler["reduce_on_plateau"] = isinstance(
                    scheduler["scheduler"], optim.lr_scheduler.ReduceLROnPlateau
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
                lr_schedulers.append({**default_config, **scheduler})
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if monitor is None:
                    raise MisconfigurationException(
                        "`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
                        " scheduler is used. For example:"
                        ' {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                    )
                lr_schedulers.append(
                    {**default_config, "scheduler": scheduler, "reduce_on_plateau": True, "monitor": monitor}
                )
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                lr_schedulers.append({**default_config, "scheduler": scheduler})
            else:
                raise ValueError(f'The provided lr scheduler "{scheduler}" is invalid')
    return lr_schedulers


def _get_default_scheduler_config() -> Dict[str, Any]:
    return {
        "scheduler": None,
        "name": None,  # no custom name
        "interval": "epoch",  # after epoch is over
        "frequency": 1,  # every epoch/batch
        "reduce_on_plateau": False,  # most often not ReduceLROnPlateau scheduler
        "monitor": None,  # value to monitor for ReduceLROnPlateau
        "strict": True,  # enforce that the monitor exists for ReduceLROnPlateau
        "opt_idx": None,  # necessary to store opt_idx when optimizer frequencies are specified
    }


def _validate_scheduler_optimizer(optimizers: List[Any], lr_schedulers: List[Any]) -> None:
    if any(sch["scheduler"].optimizer not in optimizers for sch in lr_schedulers):
        raise MisconfigurationException(
            "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
        )


def _validate_optim_conf(optim_conf: Dict[str, Any]) -> None:
    valid_keys = {"optimizer", "lr_scheduler", "frequency", "monitor"}
    extra_keys = optim_conf.keys() - valid_keys
    if extra_keys:
        rank_zero_warn(
            f"Found unsupported keys in the optimizer configuration: {set(extra_keys)}", category=RuntimeWarning
        )


def _convert_to_lightning_optimizers(trainer: "pl.Trainer") -> None:
    def _convert_to_lightning_optimizer(optimizer: Optimizer) -> LightningOptimizer:
        if not isinstance(optimizer, LightningOptimizer):
            optimizer = LightningOptimizer(optimizer)  # type: ignore [assignment]
        optimizer._on_trainer_init(trainer)
        return optimizer  # type: ignore [return-value]

    trainer._lightning_optimizers = {  # type: ignore [assignment]
        opt_idx: _convert_to_lightning_optimizer(opt) for opt_idx, opt in enumerate(trainer.optimizers)
    }


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None` is returned from
    `configure_optimizers`."""

    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    def load_state_dict(self, state_dict: Dict[Any, Any]) -> None:
        pass  # Do Nothing

    def state_dict(self) -> Dict[str, Any]:
        return {}  # Return Empty

    def step(self, closure: Callable = None) -> None:
        if closure is not None:
            closure()

    def zero_grad(self, set_to_none: Optional[bool] = False) -> None:
        pass  # Do Nothing

    def __repr__(self) -> str:
        return "No Optimizer"
