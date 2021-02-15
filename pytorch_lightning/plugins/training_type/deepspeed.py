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
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE

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
            inputs = self._move_float_tensors_to_half(inputs)

        return super().forward(*inputs, **kwargs)

    @staticmethod
    def batch_to(data):
        return data.half()

    def _move_float_tensors_to_half(self, batch: Any):
        batch = apply_to_collection(batch, (torch.FloatTensor, torch.cuda.FloatTensor), function=self.batch_to)
        return batch


class DeepSpeedPlugin(DDPPlugin):
    distributed_backend = "deepspeed"
    DEEPSPEED_ENV_VAR = "DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        config: Optional[Union[Path, str, dict]] = None,
        logging_level: int = logging.WARN,
        num_nodes: int = 1,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
    ) -> None:
        super().__init__(
            parallel_devices=parallel_devices, num_nodes=num_nodes, cluster_environment=cluster_environment
        )
        self.config = self._load_config(config)
        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

    def _load_config(self, config):
        if config is None:
            if self.DEEPSPEED_ENV_VAR not in os.environ:
                raise MisconfigurationException(
                    "You did not pass a DeepSpeed config object or path for DeepSpeed. This can be passed"
                    " via instantiating the `DeepSpeedPlugin` object, or by the DEEPSPEED_CONFIG_PATH env variable."
                    " See: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed"
                )
            config = os.environ.get(self.DEEPSPEED_ENV_VAR)
        if isinstance(config, str) or isinstance(config, Path):
            if os.path.exists(config):
                with open(config) as f:
                    config = json.load(f)
            else:
                raise MisconfigurationException(
                    f"You passed in a path to a DeepSpeed config but the path does not exist: {config}"
                )
        return config

    def pre_training(self):
        self.set_world_ranks()
        self.init_ddp_connection(self.global_rank, self.world_size)

        self.init_deepspeed()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device
        self.barrier()

    def init_deepspeed(self):
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

        precision = self.lightning_module.trainer.accelerator_backend.precision
        model = LightningDeepSpeedModule(pl_module=self.model, precision=precision)

        if self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

    def _init_scheduler_optimizer(self):
        optimizer, lightning_scheduler, optimizer_frequencies = self.lightning_module.trainer.init_optimizers(
            self.lightning_module
        )
        if (len(optimizer) != 1) or (lightning_scheduler is not None and len(lightning_scheduler) != 1):
            raise MisconfigurationException(
                "DeepSpeed currently only supports single optimizer, single optional scheduler."
            )
        lightning_scheduler = lightning_scheduler[0]['scheduler'] if lightning_scheduler else None
        optimizer = optimizer[0]
        return optimizer, lightning_scheduler, optimizer_frequencies

    def _initialize_deepspeed_train(self, model):
        optimizer, lightning_scheduler, optimizer_frequencies = None, None, None
        if "optimizer" not in self.config:
            rank_zero_info(
                "You have not specified an optimizer or scheduler within the DeepSpeed config."
                "Using `configure_optimizers` to define optimizer and scheduler."
            )
            optimizer, lightning_scheduler, optimizer_frequencies = self._init_scheduler_optimizer()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=SimpleNamespace(local_rank=self.local_rank),
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lightning_scheduler,
            config_params=self.config,
        )

        # set optimizer for save/load, but deepspeed manages the specific optimizer logic
        trainer = self.lightning_module.trainer
        trainer.optimizers = [optimizer]
        trainer.convert_to_lightning_optimizers()
        self.model = model

    def _initialize_deepspeed_inference(self, model):
        # move the model to the correct device
        self.model_to_device()

        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

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
        # Skip initializing optimizers here as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_initialize_deepspeed_train`
        return [], [], []  # empty optimizers, schedulers and frequencies

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        self.model.step(**kwargs)

    def _format_config(self):
        if not self.config:
            raise MisconfigurationException(
                "To use DeepSpeed you must pass in a DeepSpeed config dict, or a path to a JSON config."
                " See: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed"
            )
        self._format_batch_size_and_grad_accum_config()
        self._format_precision_config()

    def _format_batch_size_and_grad_accum_config(self):
        if "gradient_accumulation_steps" in self.config:
            raise MisconfigurationException(
                "Within the DeepSpeed config, do not set gradient_accumulation_steps"
                " as this will be set via accumulate_grad_batches=x argument passed via the Lightning Trainer."
            )
        if "train_micro_batch_size_per_gpu" not in self.config:
            # train_micro_batch_size_per_gpu is used for logging purposes, use loader batch size
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
