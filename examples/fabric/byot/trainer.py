from collections.abc import Mapping
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast, Literal
from lightning_utilities.core.apply_func import apply_to_collection

import torch
from lightning_utilities.core import is_overridden
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightning.fabric.fabric import _PLUGIN_INPUT, _PRECISION_INPUT, Accelerator, Fabric, Logger, Strategy, _unwrap_objects
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.types import LRScheduler, Optimizable
from lightning.pytorch.core.module import LightningModule


class Trainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: _PRECISION_INPUT = "32-true",
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        val_sanity_check: int = 2,
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
    ) -> None:

        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        # TODO: Expose as arguments
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # TODO:validation that int or float('inf')
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.val_sanity_check = val_sanity_check
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

    def fit(self, model: LightningModule, train_loader: DataLoader, val_loader: DataLoader, ckpt_path: Optional[str] = None):
        if self.fabric.is_global_zero:
            self.fabric.call("prepare_data")

        # TODO: Have fabric launch the loop function directly?
        self.fabric.launch()

        # TODO: load checkpoint if exists

        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # TODO: raise error if multiple optimizers
        # TODO: support lr scheduler
        optimizer, scheduler = self._parse_optimizers_schedulers(model.configure_optimizers())

        if isinstance(self.fabric.strategy, FSDPStrategy):
            model = self.fabric.setup_module(model)
            # FIXME: optimizer setup with wrapped model not supported through fabric with configure_optimizers
            raise NotImplementedError("BYOT currently does not support fsdp")
            optimizer = self.fabric.setup_optimizers(optimizer)
        else:
            model, optimizer = self.fabric.setup(model, optimizer)

        if self.val_sanity_check and val_loader is not None:
            self.val_loop(model, val_loader, limit_batches=self.val_sanity_check)

        # TODO: should this be a for loop?
        while not self.should_stop:
            self.train_loop(model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg)

            if self.should_eval:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(model, scheduler_cfg, level='epoch', current_value=self.current_epoch)
            # TODO: Checkpointing

            self.current_epoch += 1
            if self.max_epochs is not None and self.current_epoch > self.max_epochs:
                self.should_stop = True

    def train_loop(
        self,
        model: LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[LRScheduler, bool, str, int]]] = None,
    ):
        self.fabric.call("on_train_epoch_start")

        for batch_idx, batch in enumerate(
            self.pbar_wrapper(
                train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
            )
        ):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_train_epoch_end")
                return

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # TODO: transfer_batch_to_device_hooks don't make sense due to fabric wrapping DataLoader

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.grad_accum_steps % self.global_step == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)
                # TODO: (lower priority) support optimizer without closure?
                optimizer.step(
                    partial(self.training_step, model=model, optimizer=optimizer, batch=batch, batch_idx=batch_idx)
                )
            else:
                self.training_step(model=model, optimizer=optimizer, batch=batch, batch_idx=batch_idx)

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            self.step_scheduler(model, scheduler_cfg, level='step', current_value=self.global_step)

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True

        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self, model: LightningModule, val_loader: Optional[DataLoader], limit_batches: Union[int, float] = float("inf")
    ):
        if val_loader is None:
            return
        elif val_loader is not None and is_overridden("validation_step", _unwrap_objects(model), LightningModule):
            rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        # TODO: inference mode?
        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        for batch_idx, batch in enumerate(
            self.pbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
        ):

            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_validation_epoch_end")
                return

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        # TODO: Inference mode?
        torch.set_grad_enabled(True)

    def training_step(self, model, optimizer, batch, batch_idx):
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']

        self.fabric.call("on_before_zero_grad", optimizer)

        optimizer.zero_grad()

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")
        # TODO: reroute configure_gradient_clipping in LM to fabric.gradient clipping
        # self.fabric.call('configure_gradient_clipping')
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

    def step_scheduler(self, model: LightningModule, scheduler_cfg: Optional[Mapping[str, Union[LRScheduler, bool, str, int]]], level: Literal['step', 'epoch'], current_value: int):
        if scheduler_cfg is None:
            return

        if scheduler_cfg['interval'] != level:
            return

        if cast(int, scheduler_cfg['frequency']) % current_value != 0:
            return

        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update('train_loss', self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(self._current_train_return)

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update('val_loss', self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(self._current_val_return)

        monitor = {**self._current_train_return, **self._current_val_return}.get(scheduler_cfg['monitor'], None)
        
        model.lr_scheduler_step(scheduler_cfg['scheduler'], monitor)




    @property
    def should_validate(self) -> bool:
        return self.current_epoch % self.validation_frequency == 0

    def pbar_wrapper(self, iterable, total, **kwargs):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total, **kwargs)
        return iterable

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[Optional[Optimizable], Optional[Mapping[str, Union[LRScheduler, bool, str, int]]]]:

        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        elif isinstance(configure_optim_output, LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        elif isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        elif isinstance(configure_optim_output, (list, tuple)):
            if all([isinstance(_opt_cand, Optimizable) for _opt_cand in configure_optim_output]):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            elif all([isinstance(_lr_cand, (LRScheduler, Mapping)) for _lr_cand in configure_optim_output]):

                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None
