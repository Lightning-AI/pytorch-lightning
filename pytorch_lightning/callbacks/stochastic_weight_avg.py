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
from typing import Any, Callable, Dict, IO, List, Optional, Type, Union

import torch
from torch import nn
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class StochasticWeightAveraging(Callback):
    def __init__(
        self,
        swa_epoch_start: Union[int, float] = 0.8,
        swa_lrs: Optional[Union[float, List[float]]] = None,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        swa_validation: bool = False,
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

            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch

            swa_lrs: The SWA learning rate to use:

                - ``None``. Use the current learning rate of the optimizer at the time the SWA procedure starts.
                - ``float``. Use this value for all parameter groups of the optimizer.
                - ``List[float]``. A list values for each parameter group of the optimizer.

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

            swa_validation: if True, then the averaged model weights are used during validation
                (default: ``False``)

        """

        err_msg = "swa_epoch_start should be a >0 integer or a float between 0 and 1."
        if isinstance(swa_epoch_start, int) and swa_epoch_start < 1:
            raise MisconfigurationException(err_msg)
        if isinstance(swa_epoch_start, float) and not (0 <= swa_epoch_start <= 1):
            raise MisconfigurationException(err_msg)

        wrong_type = not isinstance(swa_lrs, (float, list))
        wrong_float = isinstance(swa_lrs, float) and swa_lrs <= 0
        wrong_list = isinstance(swa_lrs, list) and not all(lr > 0 and isinstance(lr, float) for lr in swa_lrs)
        if swa_lrs is not None and (wrong_type or wrong_float or wrong_list):
            raise MisconfigurationException(
                "The `swa_lrs` should be `None`, a positive float, or a list of positive floats"
            )

        if avg_fn is not None and not isinstance(avg_fn, Callable):
            raise MisconfigurationException("The `avg_fn` should be callable.")

        if device is not None and not isinstance(device, (torch.device, str)):
            raise MisconfigurationException(f"device is expected to be a torch.device or a str. Found {device}")

        self.n_averaged = None
        self._swa_epoch_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or self.avg_fn
        self._swa_validation = swa_validation
        self._device = device
        self._model_contains_batch_norm = None
        self._average_model = None
        self._temp_model = None
        self._initialized = False
        self._swa_scheduler = None
        self._batch_norm_moments = None
        self._scheduler_step_count = None

    @property
    def swa_start(self) -> int:
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule"):
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # copy the model before moving it to accelerator device.
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._average_model = deepcopy(pl_module)
            if self._swa_validation:
                # Also create a model for temporarily copying weights to during validation
                self._temp_model = deepcopy(pl_module)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers

        if len(optimizers) != 1:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if len(lr_schedulers) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 `lr_scheduler`.")

        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        resuming_after_start = (not self._initialized) and (self.swa_start < trainer.current_epoch <= self.swa_end)
        if trainer.current_epoch == self.swa_start or resuming_after_start:
            self._initialized = True

            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)
            if self._temp_model:
                self._temp_model = self._temp_model.to(self._device or pl_module.device)

            optimizer = trainer.optimizers[0]
            if self._swa_lrs is None:
                self._swa_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)

            for lr, group in zip(self._swa_lrs, optimizer.param_groups):
                group["initial_lr"] = lr

            self._swa_scheduler = SWALR(
                optimizer,
                swa_lr=self._swa_lrs,
                anneal_epochs=self._annealing_epochs,
                anneal_strategy=self._annealing_strategy,
                last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
            )
            if self._scheduler_step_count is not None:
                # Restore scheduler step count from checkpoint
                self._swa_scheduler._step_count = self._scheduler_step_count
            default_scheduler_cfg = _get_default_scheduler_config()
            assert default_scheduler_cfg["interval"] == "epoch" and default_scheduler_cfg["frequency"] == 1
            default_scheduler_cfg["scheduler"] = self._swa_scheduler

            if trainer.lr_schedulers:
                scheduler_cfg = trainer.lr_schedulers[0]
                if scheduler_cfg["interval"] != "epoch" or scheduler_cfg["frequency"] != 1:
                    rank_zero_warn(f"SWA is currently only supported every epoch. Found {scheduler_cfg}")
                rank_zero_info(
                    f"Swapping scheduler `{scheduler_cfg['scheduler'].__class__.__name__}`"
                    f" for `{self._swa_scheduler.__class__.__name__}`"
                )
                trainer.lr_schedulers[0] = default_scheduler_cfg
            else:
                trainer.lr_schedulers.append(default_scheduler_cfg)

            if self.n_averaged is None:
                self.n_averaged = torch.tensor(0, dtype=torch.long, device=pl_module.device)

        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self.avg_fn)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch == self.swa_end:
            # Last SWA epoch. Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model, pl_module)
            if self._model_contains_batch_norm:
                self._update_batch_norm_moments(trainer, pl_module, store_moments=False)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._swa_validation and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            # Take a temporary copy of the model parameters
            self.transfer_weights(pl_module, self._temp_model)
            # Update the model with the averaged parameters
            self.transfer_weights(self._average_model, pl_module)
            if self._model_contains_batch_norm:
                self._update_batch_norm_moments(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._swa_validation and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            # Copy original model parameters back
            self.transfer_weights(self._temp_model, pl_module)
            if self._model_contains_batch_norm:
                self._restore_batch_norm_moments()

    @staticmethod
    def transfer_weights(src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule"):
        for src_param, dst_param in zip(src_pl_module.parameters(), dst_pl_module.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def _update_batch_norm_moments(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", store_moments: bool = True
    ):
        self._batch_norm_moments = {}

        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            # Training data not yet connected, could be in a validation sanity check
            return

        self._update_module_batch_norm_moments(
            train_dataloader, pl_module, self._batch_norm_moments if store_moments else None
        )

    @staticmethod
    def _update_module_batch_norm_moments(
        data_loader: Union[DataLoader, CombinedLoader],
        pl_module: "pl.LightningModule",
        moment_cache: Optional[Dict[nn.Module, Any]] = None,
    ):
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L166."""
        prev_momenta = {}

        was_training = pl_module.training
        pl_module.train()

        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            prev_momenta[module] = module.momentum
            if moment_cache is not None:
                moment_cache[module] = (module.running_mean, module.running_var)
            module.running_mean = torch.zeros_like(
                module.running_mean, device=pl_module.device, dtype=module.running_mean.dtype
            )
            module.running_var = torch.ones_like(
                module.running_var, device=pl_module.device, dtype=module.running_var.dtype
            )
            module.momentum = None
            module.num_batches_tracked *= 0

        # Recompute mean and variance for all batch norm layers by doing a full pass over the training data
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(pl_module.device)
            pl_module(batch)

        # Reset model state
        for bn_module, momenta in prev_momenta.items():
            bn_module.momentum = momenta
        pl_module.train(was_training)

    def _restore_batch_norm_moments(self):
        for bn_module, (mean, variance) in self._batch_norm_moments.items():
            bn_module.running_mean = mean
            bn_module.running_var = variance

    @staticmethod
    def update_parameters(
        average_model: "pl.LightningModule", model: "pl.LightningModule", n_averaged: torch.LongTensor, avg_fn: _AVG_FN
    ):
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112."""
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = p_model_ if n_averaged == 0 else avg_fn(p_swa_, p_model_, n_averaged.to(device))
            p_swa_.copy_(src)
        n_averaged += 1

    @staticmethod
    def avg_fn(
        averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
    ) -> torch.FloatTensor:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> dict:
        checkpoint_data = {
            "n_averaged": self.n_averaged,
            "swa_lrs": self._swa_lrs,
            "annealing_epochs": self._annealing_epochs,
            "annealing_strategy": self._annealing_strategy,
            "scheduler_step_count": None if self._swa_scheduler is None else self._swa_scheduler._step_count,
            "average_model_parameters": self._get_average_model_parameters(trainer),
        }
        return checkpoint_data

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        if callback_state:
            self.n_averaged = callback_state["n_averaged"]
            self._swa_lrs = callback_state["swa_lrs"]
            self._annealing_strategy = callback_state["annealing_strategy"]
            self._annealing_epochs = callback_state["annealing_epochs"]
            self._scheduler_step_count = callback_state["scheduler_step_count"]
            self._load_average_model_parameters(callback_state["average_model_parameters"])
        else:
            rank_zero_warn(
                f"Checkpoint has no data for the {self.state_key} callback, not initializing the callback state."
            )

    @classmethod
    def restore_average_parameters_from_checkpoint(
        cls,
        pl_module: "pl.LightningModule",
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> bool:
        r"""
        Set model weights to the SWA averaged weights saved in a checkpoint.

        When loading a model that was trained using SWA from a checkpoint,
        the loaded weights will not be the SWA averaged weights, so this method is required if you
        wish to use SWA in conjunction with the :class:`~pytorch_lightning.callbacks.ModelCheckpoint`
        callback to select the best performing model during validation for example.

        Arguments:
            pl_module: The module to set weights on

            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object

            map_location: If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.

            datamodule: If the module uses batch normalization and does not implement the ``train_dataloder`` method,
                a data module must be provided in order to allow recomputing the batch normalization parameters after
                loading the SWA weights.

        Return:
            Whether averaged weights were loaded. If ``False``, this means the checkpoint is
            from an epoch before the SWA epoch start.
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        callback_states: Dict[Union[Type, str], Dict] = checkpoint.get("callbacks")
        if not callback_states:
            raise ValueError("callback states are not present in the checkpoint")

        state_key = cls.__qualname__  # Default state key defined in Callback base class
        state = callback_states.get(state_key)
        if not state:
            raise ValueError(f"no {state_key} state found in the checkpoint")
        state = deepcopy(state)
        average_model_parameters = state["average_model_parameters"]

        if not average_model_parameters:
            return False

        for p_model, p_swa in zip(pl_module.parameters(), average_model_parameters):
            device = p_model.device
            p_swa_ = p_swa.detach().to(device)
            p_model.detach().copy_(p_swa_)

        if cls.pl_module_contains_batch_norm(pl_module):
            if datamodule is not None:
                train_dataloaders = datamodule.train_dataloader()
            else:
                train_dataloaders = pl_module.train_dataloader()
            train_dataloaders = CombinedLoader(train_dataloaders, mode="max_size_cycle")
            cls._update_module_batch_norm_moments(train_dataloaders, pl_module)

        return True

    def _get_average_model_parameters(self, trainer: "pl.Trainer") -> Any:
        if self._average_model is None or not (self.swa_start <= trainer.current_epoch <= self.swa_end):
            # If we're not within the SWA epochs then when loading checkpoint data we would want
            # to use parameters from the underlying model rather than the SWA parameters.
            return None
        parameters = []
        for p_swa in self._average_model.parameters():
            parameters.append(p_swa.detach())
        return parameters

    def _load_average_model_parameters(self, parameter_state: Any):
        if self._average_model is None:
            return
        for p_swa, p_checkpoint in zip(self._average_model.parameters(), parameter_state):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_checkpoint_ = p_checkpoint.detach().to(device)
            p_swa_.copy_(p_checkpoint_)
