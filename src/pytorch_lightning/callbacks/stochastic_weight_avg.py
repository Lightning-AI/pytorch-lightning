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
Stochastic Weight Averaging Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
from copy import deepcopy
from typing import Any, Callable, cast, Dict, List, Optional, Union

import torch
from torch import nn, Tensor
from torch.optim.swa_utils import SWALR

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.strategies import DDPFullyShardedStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import _LRScheduler, LRSchedulerConfig

_AVG_FN = Callable[[Tensor, Tensor, Tensor], Tensor]


class StochasticWeightAveraging(Callback):
    def __init__(
        self,
        swa_lrs: Union[float, List[float]],
        swa_epoch_start: Union[int, float] = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
    ):
        r"""

        Implements the Stochastic Weight Averaging (SWA) Callback to average a model.

        Stochastic Weight Averaging was proposed in ``Averaging Weights Leads to
        Wider Optima and Better Generalization`` by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        This documentation is highly inspired by PyTorch's work on SWA.
        The callback arguments follow the scheme defined in PyTorch's ``swa_utils`` package.

        For a SWA explanation, please take a look
        `here <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`_.

        .. warning:: ``StochasticWeightAveraging`` is in beta and subject to change.

        .. warning:: ``StochasticWeightAveraging`` is currently not supported for multiple optimizers/schedulers.

        .. warning:: ``StochasticWeightAveraging`` is currently only supported on every epoch.

        See also how to :ref:`enable it directly on the Trainer <advanced/training_tricks:Stochastic Weight Averaging>`

        Arguments:

            swa_lrs: The SWA learning rate to use:

                - ``float``. Use this value for all parameter groups of the optimizer.
                - ``List[float]``. A list values for each parameter group of the optimizer.

            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch

            annealing_epochs: number of epochs in the annealing phase (default: 10)

            annealing_strategy: Specifies the annealing strategy (default: "cos"):

                - ``"cos"``. For cosine annealing.
                - ``"linear"`` For linear annealing

            avg_fn: the averaging function used to update the parameters;
                the function must take in the current value of the
                :class:`AveragedModel` parameter, the current value of :attr:`model`
                parameter and the number of models already averaged; if None,
                equally weighted average is used (default: ``None``)

            device: if provided, the averaged model will be stored on the ``device``.
                When None is provided, it will infer the `device` from ``pl_module``.
                (default: ``"cpu"``)

        """

        err_msg = "swa_epoch_start should be a >0 integer or a float between 0 and 1."
        if isinstance(swa_epoch_start, int) and swa_epoch_start < 1:
            raise MisconfigurationException(err_msg)
        if isinstance(swa_epoch_start, float) and not (0 <= swa_epoch_start <= 1):
            raise MisconfigurationException(err_msg)

        wrong_type = not isinstance(swa_lrs, (float, list))
        wrong_float = isinstance(swa_lrs, float) and swa_lrs <= 0
        wrong_list = isinstance(swa_lrs, list) and not all(lr > 0 and isinstance(lr, float) for lr in swa_lrs)
        if wrong_type or wrong_float or wrong_list:
            raise MisconfigurationException("The `swa_lrs` should a positive float, or a list of positive floats")

        if avg_fn is not None and not callable(avg_fn):
            raise MisconfigurationException("The `avg_fn` should be callable.")

        if device is not None and not isinstance(device, (torch.device, str)):
            raise MisconfigurationException(f"device is expected to be a torch.device or a str. Found {device}")

        self.n_averaged: Optional[torch.Tensor] = None
        self._swa_epoch_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or self.avg_fn
        self._device = device
        self._model_contains_batch_norm: Optional[bool] = None
        self._average_model: "pl.LightningModule"
        self._initialized = False
        self._swa_scheduler: Optional[_LRScheduler] = None
        self._scheduler_state: Optional[Dict] = None
        self._init_n_averaged = 0
        self._latest_update_epoch = -1
        self.momenta: Optional[Dict[nn.modules.batchnorm._BatchNorm, float]] = None
        self._max_epochs: int

    @property
    def swa_start(self) -> int:
        assert isinstance(self._swa_epoch_start, int)
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule") -> bool:
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        # copy the model before moving it to accelerator device.
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._average_model = deepcopy(pl_module)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(trainer.optimizers) != 1:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if len(trainer.lr_scheduler_configs) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 `lr_scheduler`.")

        if isinstance(trainer.strategy, (DDPFullyShardedStrategy, DeepSpeedStrategy)):
            raise MisconfigurationException("SWA does not currently support sharded models.")

        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs
        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            assert trainer.fit_loop.max_epochs is not None
            trainer.fit_loop.max_epochs += 1

        if self._scheduler_state is not None:
            self._clear_schedulers(trainer)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            self._initialized = True

            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

            optimizer = trainer.optimizers[0]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)

            for lr, group in zip(self._swa_lrs, optimizer.param_groups):
                group["initial_lr"] = lr

            self._swa_scheduler = cast(
                _LRScheduler,
                SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,  # type: ignore[arg-type]
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
                ),
            )
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif trainer.current_epoch != self.swa_start:
                # Log a warning if we're initializing after start without any checkpoint data,
                # as behaviour will be different compared to having checkpoint data.
                rank_zero_warn(
                    "SWA is initializing after swa_start without any checkpoint data. "
                    "This may be caused by loading a checkpoint from an older version of PyTorch Lightning."
                )

            # We assert that there is only one optimizer on fit start, so know opt_idx is always 0
            default_scheduler_cfg = LRSchedulerConfig(self._swa_scheduler, opt_idx=0)
            assert default_scheduler_cfg.interval == "epoch" and default_scheduler_cfg.frequency == 1

            if trainer.lr_scheduler_configs:
                scheduler_cfg = trainer.lr_scheduler_configs[0]
                if scheduler_cfg.interval != "epoch" or scheduler_cfg.frequency != 1:
                    rank_zero_warn(f"SWA is currently only supported every epoch. Found {scheduler_cfg}")
                rank_zero_info(
                    f"Swapping scheduler `{scheduler_cfg.scheduler.__class__.__name__}`"
                    f" for `{self._swa_scheduler.__class__.__name__}`"
                )
                trainer.lr_scheduler_configs[0] = default_scheduler_cfg
            else:
                trainer.lr_scheduler_configs.append(default_scheduler_cfg)

            if self.n_averaged is None:
                self.n_averaged = torch.tensor(self._init_n_averaged, dtype=torch.long, device=pl_module.device)

        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (
            trainer.current_epoch > self._latest_update_epoch
        ):
            assert self.n_averaged is not None
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self._avg_fn)
            self._latest_update_epoch = trainer.current_epoch

        # Note: No > here in case the callback is saved with the model and training continues
        if trainer.current_epoch == self.swa_end + 1:
            # Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model, pl_module)

            # Reset BatchNorm for update
            self.reset_batch_norm_and_save_state(pl_module)

            # There is no need to perform either backward or optimizer.step as we are
            # performing only one pass over the train data-loader to compute activation statistics
            # Therefore, we will virtually increase `num_training_batches` by 1 and skip backward.
            trainer.num_training_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches

            trainer.accumulate_grad_batches = trainer.num_training_batches

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args: Any) -> None:
        trainer.fit_loop._skip_backward = False

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # the trainer increases the current epoch before this hook is called
        if self._model_contains_batch_norm and trainer.current_epoch - 1 == self.swa_end + 1:
            # BatchNorm epoch update. Reset state
            trainer.accumulate_grad_batches = self._accumulate_grad_batches
            trainer.num_training_batches -= 1
            assert trainer.fit_loop.max_epochs is not None
            trainer.fit_loop.max_epochs -= 1
            self.reset_momenta()
        elif trainer.current_epoch - 1 == self.swa_end:
            # Last SWA epoch. Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model, pl_module)

    @staticmethod
    def transfer_weights(src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule") -> None:
        for src_param, dst_param in zip(src_pl_module.parameters(), dst_pl_module.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def reset_batch_norm_and_save_state(self, pl_module: "pl.LightningModule") -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154."""
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            assert module.running_mean is not None
            module.running_mean = torch.zeros_like(
                module.running_mean,
                device=pl_module.device,
                dtype=module.running_mean.dtype,
            )
            assert module.running_var is not None
            module.running_var = torch.ones_like(
                module.running_var,
                device=pl_module.device,
                dtype=module.running_var.dtype,
            )
            self.momenta[module] = module.momentum
            module.momentum = float()
            assert module.num_batches_tracked is not None
            module.num_batches_tracked *= 0

    def reset_momenta(self) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165."""
        assert self.momenta is not None
        for bn_module in self.momenta:
            bn_module.momentum = self.momenta[bn_module]

    @staticmethod
    def update_parameters(
        average_model: "pl.LightningModule", model: "pl.LightningModule", n_averaged: Tensor, avg_fn: _AVG_FN
    ) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112."""
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = p_model_ if n_averaged == 0 else avg_fn(p_swa_, p_model_, n_averaged.to(device))
            p_swa_.copy_(src)
        n_averaged += 1

    @staticmethod
    def avg_fn(averaged_model_parameter: Tensor, model_parameter: Tensor, num_averaged: Tensor) -> Tensor:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_averaged": 0 if self.n_averaged is None else self.n_averaged.item(),
            "latest_update_epoch": self._latest_update_epoch,
            "scheduler_state": None if self._swa_scheduler is None else self._swa_scheduler.state_dict(),
            "average_model_state": None if self._average_model is None else self._average_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._init_n_averaged = state_dict["n_averaged"]
        self._latest_update_epoch = state_dict["latest_update_epoch"]
        self._scheduler_state = state_dict["scheduler_state"]
        self._load_average_model_state(state_dict["average_model_state"])

    @staticmethod
    def _clear_schedulers(trainer: "pl.Trainer") -> None:
        # If we have scheduler state saved, clear the scheduler configs so that we don't try to
        # load state into the wrong type of schedulers when restoring scheduler checkpoint state.
        # We'll configure the scheduler and re-load its state in on_train_epoch_start.
        # Note that this relies on the callback state being restored before the scheduler state is
        # restored, and doesn't work if restore_checkpoint_after_setup is True, but at the time of
        # writing that is only True for deepspeed which is already not supported by SWA.
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/11665 for background.
        if trainer.lr_scheduler_configs:
            assert len(trainer.lr_scheduler_configs) == 1
            trainer.lr_scheduler_configs.clear()

    def _load_average_model_state(self, model_state: Any) -> None:
        if self._average_model is None:
            return
        self._average_model.load_state_dict(model_state)
