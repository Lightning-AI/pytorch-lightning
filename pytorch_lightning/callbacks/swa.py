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

"""
Stochastic Weight Averaging Callback
====================================

"""
from typing import Callable, Optional, Union

import torch
from torch import nn

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import _PYTORCH_GREATER_EQUAL_1_6_0, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _PYTORCH_GREATER_EQUAL_1_6_0:
    from torch.optim.swa_utils import AveragedModel, SWALR

    class LightningAveragedModel(AveragedModel):

        def __init__(self, pl_module, *args, **kwargs):
            for k, v in vars(pl_module).items():
                setattr(self, k, v)

            for fn_name in dir(pl_module):
                if not fn_name.startswith("__"):
                    setattr(self, fn_name, getattr(pl_module, fn_name))

            super().__init__(pl_module, *args, **kwargs)


class StochasticWeightAveragingCallback(Callback):

    def __init__(
        self,
        swa_epoch_start: int = 0,
        swa_lrs: Optional[Union[float, list]] = None,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[Callable] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):

        r"""Implements averaged model for Stochastic Weight Averaging (SWA) Callbacks.

        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        AveragedModel class creates a copy of the provided module :attr:`model`
        on the device :attr:`device` and allows to compute running averages of the
        parameters of the :attr:`model`.

        Arguments:

            swa_epoch_start (int): If provided, the average model will start from
                ``swa_epoch_start`` epoch

            swa_lrs (float or list): the learning rate value for all param groups
                together or separately for each group.

            annealing_epochs (int): number of epochs in the annealing phase
                (default: 10)

            annealing_strategy (str): "cos" or "linear"; specifies the annealing
                strategy: "cos" for cosine annealing, "linear" for linear annealing
                (default: "cos")

            device (torch.device, optional): if provided, the averaged model will be
                stored on the `device`. Default: `cpu`

        """

        if not isinstance(swa_epoch_start, int) or isinstance(swa_epoch_start, int) and swa_epoch_start < 0:
            raise MisconfigurationException("swa_epoch_start should be positive integer.")

        if not isinstance(swa_lrs, (float, list)) \
           or isinstance(swa_lrs, float) and swa_lrs <= 0 \
           or isinstance(swa_lrs, list) and not all(lr > 0 for lr in swa_lrs):
            raise MisconfigurationException("swa_lrs should be a positive float or a list of positive float.")

        if isinstance(avg_fn, Callable):
            raise MisconfigurationException("avg_fn should be function.")

        self._swa_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn
        self._device = device
        self._model_contains_batch_norm = None

    @property
    def swa_model(self):
        return getattr(self, "_average_model")

    @staticmethod
    def pl_module_contains_batch_norm(pl_module):
        for module in pl_module.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                return True
        return False

    def reset_batch_norm_and_save_state(self, average_model, device):
        self.momenta = {}
        for module in average_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                running_mean_dtype = module.running_mean.dtype
                running_var_dype = module.running_var.dtype
                module.running_mean = torch.zeros_like(module.running_mean, device=device, dtype=running_mean_dtype)
                module.running_var = torch.ones_like(module.running_var, device=device, dtype=running_var_dype)
                self.momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

    def apply_momemta(self):
        for bn_module in self.momenta.keys():
            bn_module.momentum = self.momenta[bn_module]

    def on_fit_start(self, trainer, pl_module):
        self._average_model = LightningAveragedModel(pl_module, device=self._device, avg_fn=self._avg_fn)
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers

        if len(optimizers) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 optimizer.")

        if len(lr_schedulers) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 lr_scheduler.")

        self._max_epochs = trainer.max_epochs

        self._swa_scheduler = SWALR(
            optimizers[0],
            swa_lr=self._swa_lrs,
            anneal_epochs=self._annealing_epochs,
            anneal_strategy=self._annealing_strategy,
            last_epoch=self._max_epochs if self._annealing_strategy == "cos" else -1
        )

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update
            trainer.max_epochs += 1

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self._swa_start:
            optimizers = trainer.optimizers
            lr_scheduler = trainer.lr_schedulers[0]["scheduler"]

            swa_scheduler = SWALR(
                optimizers[0],
                swa_lr=self._swa_lrs,
                anneal_epochs=self._annealing_epochs,
                anneal_strategy=self._annealing_strategy,
                last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1
            )

            rank_zero_warn(f"swapping lr_scheduler {lr_scheduler} for {self._swa_scheduler}")

            trainer.lr_schedulers[0]["scheduler"] = swa_scheduler

        elif self._model_contains_batch_norm and trainer.current_epoch == self._max_epochs:
            trainer.train_loop.do_backward = False

            # save curent model
            device = pl_module.device
            self._pl_module = pl_module.to("cpu")

            self._average_model = self._average_model.to(device)
            self._average_model._results = self._pl_module._results
            self.reset_batch_norm_and_save_state(self._average_model, device)
            trainer.model_connector.set_model(self._average_model)

            # perform accumulation
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
            trainer.accumulate_grad_batches = len(trainer.train_dataloader)
            trainer.train_loop.do_backward = False

    def on_train_epoch_end(self, trainer, pl_module, *_):
        if self._model_contains_batch_norm and trainer.current_epoch == self._max_epochs:
            trainer.model_connector.set_model(self._pl_module)

    def on_train_end(self, trainer, pl_module):
        trainer.accumulate_grad_batches = self._accumulate_grad_batches
        self.apply_momemta()
