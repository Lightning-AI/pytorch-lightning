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

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple, Union

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.apply_func import move_float_tensors_to_half
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.seed import seed_everything

if _DEEPSPEED_AVAILABLE:
    import deepspeed
else:
    deepspeed = None


class LightningDeepSpeedModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule, precision: int):
        super().__init__(pl_module)
        self.module = pl_module
        self.precision = precision

    def forward(self, *inputs, **kwargs):
        if self.precision == 16:
            inputs = move_float_tensors_to_half(inputs)
        return super().forward(*inputs, **kwargs)


class DeepSpeedPlugin(DDPPlugin):
    distributed_backend = "deepspeed"

    def __init__(
        self,
        config: Union[Path, str, dict],
        logging_level: int = logging.WARN,
        num_nodes=1,
        parallel_devices: List[torch.device] = None,
        cluster_environment: ClusterEnvironment = None,
    ) -> None:
        super().__init__(
            parallel_devices=parallel_devices, num_nodes=num_nodes, cluster_environment=cluster_environment
        )
        if isinstance(config, str) or isinstance(config, Path):
            with open(config) as f:
                self.config = json.load(f)
        else:
            self.config = config
        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

    def pre_training(self):
        self.set_world_ranks()
        self.init_ddp_connection(self.global_rank, self.world_size)

        self.init_deepspeed()

        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        # move the model to the correct device
        self.model_to_device()
        self.barrier()

    def init_deepspeed(self):
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

        precision = self.lightning_module.trainer.accelerator_backend.precision
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=SimpleNamespace(local_rank=self.local_rank),
            model=LightningDeepSpeedModule(pl_module=self.model, precision=precision),
            model_parameters=model_parameters,
            config_params=self.config,
        )
        trainer = self.lightning_module.trainer
        if self.lightning_module.training:
            trainer.optimizers = [optimizer]
            trainer.lr_schedulers = self.configure_scheduler(lr_scheduler)
            trainer.convert_to_lightning_optimizers()
        self.model = model

    def configure_scheduler(self, lr_scheduler):
        # this duplicates the defaults from init_optimizers
        scheduler = {
            'scheduler': lr_scheduler,
            'name': None,  # no custom name
            'interval': 'epoch',  # after epoch is over
            'frequency': 1,  # every epoch/batch
            'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
            'monitor': None,  # value to monitor for ReduceLROnPlateau
            'strict': True,  # enforce that the monitor exists for ReduceLROnPlateau
        }
        return [scheduler]

    @property
    def lightning_module(self):
        # the model may not be wrapped with DeepEngine & LightningDeepSpeedModule if calling this too early
        module = getattr(self.model, "module", self.model)
        return module.module if isinstance(module, LightningDeepSpeedModule) else module

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=self.world_size, rank=self.global_rank)
        return distributed_sampler_kwargs

    def init_optimizers(self, trainer: "Trainer", model: LightningModule) -> Tuple[List, List, List]:
        # Skip initializing optimizers as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_format_config`
        return [], [], []  # empty optimizers, schedulers and frequencies

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        self.model.step(**kwargs)

    def _format_config(self):
        if not self.config:
            raise MisconfigurationException(
                "To use DeepSpeed you must pass in a DeepSpeed config dictionary, or path to a json config."
                "todo: Doc Link."
            )
        self._format_optimizer_config()
        self._format_batch_size_grad_accum_config()
        self._format_precision_config()

    def _format_optimizer_config(self):
        if ("optimizer" not in self.config) or ("scheduler" not in self.config):
            self.optimizer, self.scheduler = self.model.configure_optimizers()

            if not (isinstance(self.optimizer, dict) or isinstance(self.scheduler, dict)):
                raise MisconfigurationException(
                    "If you have not specified an optimizer or scheduler within the DeepSpeed config "
                    "then you must return a dict from `configure_optimizers` within the LightningModule. "
                    "See x for more information."
                )

            if not len(self.optimizer) == 1 or len(self.scheduler) == 1:
                raise MisconfigurationException("DeepSpeed currently only supports single optimizer, single scheduler.")

            optimizer_name, optimizer_params = self.optimizer.items()[0]
            scheduler_name, scheduler_params = self.scheduler.items()[0]

            self.config["optimizer"] = {
                "type": optimizer_name,
                "params": optimizer_params,
            }
            self.config["scheduler"] = {
                "type": scheduler_name,
                "params": scheduler_params,
            }

    def _format_batch_size_grad_accum_config(self):
        if "train_batch_size" in self.config or "train_micro_batch_size_per_gpu" in self.config:
            raise MisconfigurationException(
                "Within the DeepSpeed config, do not set train_batch_size or train_micro_batch_size_per_gpu "
                "as these will be passed from the data-loader."
            )
        if "gradient_accumulation_steps" in self.config:
            raise MisconfigurationException(
                "Within the DeepSpeed config, do not set gradient_accumulation_steps "
                "as this will be set via accumulate_grad_batches=x argument passed via the Lightning Trainer."
            )
        self.config["train_micro_batch_size_per_gpu"] = self.lightning_module.train_dataloader().batch_size
        self.config["gradient_accumulation_steps"] = self.lightning_module.trainer.accumulate_grad_batches
        if "gradient_clipping" not in self.config:
            self.config["gradient_clipping"] = self.lightning_module.trainer.gradient_clip_val

    def _format_precision_config(self):

        amp_type = self.lightning_module.trainer.accelerator_connector.amp_type
        amp_level = self.lightning_module.trainer.accelerator_connector.amp_level
        precision = self.lightning_module.trainer.accelerator_connector.precision
        if precision == 16:
            if "amp" not in self.config and amp_type == AMPType.NATIVE:
                self.config["fp16"] = {"enabled": True}
            elif "apex" not in self.config and amp_type == AMPType.APEX:
                self.config["amp"] = {
                    "enabled": True,
                    "opt_level": amp_level,
                }
