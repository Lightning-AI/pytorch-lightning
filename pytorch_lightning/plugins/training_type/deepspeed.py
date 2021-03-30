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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE

if _DEEPSPEED_AVAILABLE:
    import deepspeed


class LightningDeepSpeedModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule, precision: int):
        super().__init__(pl_module)
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
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        zero_optimization: bool = True,
        stage: int = 2,
        cpu_offload: bool = False,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 2e8,
        reduce_bucket_size: int = 2e8,
        zero_allow_untested_optimizer: bool = True,
        config: Optional[Union[Path, str, dict]] = None,
        logging_level: int = logging.WARN,
        num_nodes: int = 1,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 32,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1
    ) -> None:
        """

        Provides capabilities to run training using the DeepSpeed library,
        with training optimizations for large billion parameter models.
        `For more information: https://www.deepspeed.ai/`.

        .. warning:: ``DeepSpeedPlugin`` is in beta and subject to change.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is only compatible with precision=16. (default: True)

            stage: Different stages of the ZeRO Optimizer. 0 is disabled,
                1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning (default: 2)

            cpu_offload: Enable offloading optimizer memory and computation to CPU

            contiguous_gradients: Copies gradients to a continuous buffer as they are produced.
                Avoids memory fragmentation during backwards. Useful when training large models. (default: True)

            overlap_comm: Overlap the reduction (synchronization) of gradients with the backwards computation.
                This is a speed optimization when training across multiple GPUs/machines. (default: True)

            allgather_partitions: All gather updated parameters at the end of training step,
                instead of using a series of broadcast collectives (default: True)

            reduce_scatter: Use reduce/scatter instead of allreduce to average gradients (default:True)

            allgather_bucket_size: Number of elements to allgather at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed. (default: 2e8)

            reduce_bucket_size: Number of elements to reduce at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed (default: 2e8)

            zero_allow_untested_optimizer: Allow untested optimizers to be used with ZeRO. Currently only Adam is a
                DeepSpeed supported optimizer when using ZeRO (default: True)

            config: Pass in a deepspeed formatted config dict,
                or path to a deepspeed config: https://www.deepspeed.ai/docs/config-json.
                All defaults will be ignored if a config is passed in. (Default: ``None``)

            logging_level: Set logging level for deepspeed. (Default: ``logging.WARN``)

            loss_scale: Loss scaling value for FP16 training.
                0.0 results in dynamic loss scaling, otherwise static (Default: 0)

            initial_scale_power: Power of the initial dynamic loss scale value. Loss scale is computed
                by ``2^initial_scale_power`` (Default: 32)

            loss_scale_window: Window in which to raise/lower the dynamic FP16 loss scaling value (Default: 1000)

            hysteresis: FP16 Delay shift in Dynamic Loss scaling (Default: 2)

            min_loss_scale: The minimum FP16 dynamic loss scaling value (Default: 1000)

        """
        if not _DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the DeepSpeed plugin, you must have DeepSpeed installed."
                " pip install deepspeed mpi4py"
            )
        super().__init__(
            parallel_devices=parallel_devices, num_nodes=num_nodes, cluster_environment=cluster_environment
        )
        self.config = self._load_config(config)
        if self.config is None:
            # User has not overridden config, set defaults
            self.config = self._create_default_config(
                zero_optimization,
                zero_allow_untested_optimizer,
                stage=stage,
                cpu_offload=cpu_offload,
                contiguous_gradients=contiguous_gradients,
                overlap_comm=overlap_comm,
                allgather_partitions=allgather_partitions,
                reduce_scatter=reduce_scatter,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size
            )
        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

        # default FP16 parameters.
        self.loss_scale = loss_scale
        self.initial_scale_power = initial_scale_power
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale

    def _load_config(self, config):
        if config is None and self.DEEPSPEED_ENV_VAR in os.environ:
            rank_zero_info(f"Loading DeepSpeed config from set {self.DEEPSPEED_ENV_VAR} environment variable")
            config = os.environ[self.DEEPSPEED_ENV_VAR]
        if isinstance(config, str) or isinstance(config, Path):
            if not os.path.isfile(config):
                raise MisconfigurationException(
                    f"You passed in a path to a DeepSpeed config but the path does not exist: {config}"
                )
            with open(config) as f:
                config = json.load(f)
        return config

    def pre_dispatch(self):
        self.init_deepspeed()
        self.barrier()

    def init_deepspeed(self):
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

        precision = self.lightning_module.trainer.accelerator.precision
        model = LightningDeepSpeedModule(pl_module=self.model, precision=precision)

        if self.lightning_module.trainer and self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

    def _init_scheduler_optimizer(self):
        optimizers, schedulers, optimizer_frequencies = self.lightning_module.trainer.init_optimizers(
            self.lightning_module
        )
        if len(optimizers) > 1 or len(schedulers) > 1:
            raise MisconfigurationException(
                "DeepSpeed currently only supports single optimizer, single optional scheduler."
            )
        scheduler = schedulers[0]['scheduler'] if len(schedulers) == 1 else None
        optimizer = optimizers[0]
        return optimizer, scheduler, optimizer_frequencies

    def _initialize_deepspeed_train(self, model):
        if self.on_gpu:
            torch.cuda.set_device(self.root_device)
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
        self.lightning_module.trainer.optimizers = [optimizer]
        self.model = model

    def _initialize_deepspeed_inference(self, model):
        # move the model to the correct device
        self.model_to_device()

        self.pre_configure_ddp()
        self.model = DistributedDataParallel(
            model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def configure_scheduler(self, lr_scheduler):
        scheduler = _get_default_scheduler_config()
        scheduler["scheduler"] = lr_scheduler
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

    def init_optimizers(self, trainer, model: LightningModule) -> Tuple[List, List, List]:
        # Skip initializing optimizers here as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_initialize_deepspeed_train`
        return [], [], []  # empty optimizers, schedulers and frequencies

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        # note: We rely on the deepspeed engine to carry out the step rather than the optimizer.
        # internally, the engine has a reference to the optimizer already.
        self.model.step(**kwargs)

    def _format_config(self):
        if self.config is None:
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
            # train_micro_batch_size_per_gpu is used for throughput logging purposes
            # by default we use the batch size of the loader which may be incorrect if a batch sampler is passed
            batch_size = self.lightning_module.train_dataloader().batch_size
            self.config["train_micro_batch_size_per_gpu"] = batch_size
        self.config["gradient_accumulation_steps"] = self.lightning_module.trainer.accumulate_grad_batches
        if "gradient_clipping" not in self.config:
            self.config["gradient_clipping"] = self.lightning_module.trainer.gradient_clip_val

    def _format_precision_config(self):

        amp_type = self.lightning_module.trainer.accelerator_connector.amp_type
        amp_level = self.lightning_module.trainer.accelerator_connector.amp_level
        precision = self.lightning_module.trainer.accelerator_connector.precision
        if precision == 16:
            if "fp16" not in self.config and amp_type == AMPType.NATIVE:
                # FP16 is a DeepSpeed standalone AMP implementation
                rank_zero_info("Enabling DeepSpeed FP16.")
                self.config["fp16"] = {
                    "enabled": True,
                    "loss_scale": self.loss_scale,
                    "initial_scale_power": self.initial_scale_power,
                    "loss_scale_window": self.loss_scale_window,
                    "hysteresis": self.hysteresis,
                    "min_loss_scale": self.min_loss_scale
                }
            elif "amp" not in self.config and amp_type == AMPType.APEX:
                rank_zero_only("Enabling DeepSpeed APEX Implementation.")
                self.config["amp"] = {
                    "enabled": True,
                    "opt_level": amp_level,
                }
        if "zero_optimization" in self.config and not ("amp" in self.config or "fp16" in self.config):
            raise MisconfigurationException("To use DeepSpeed ZeRO Optimization, you must set precision=16.")

    def _create_default_config(
        self, zero_optimization: bool, zero_allow_untested_optimizer: bool, **zero_kwargs
    ) -> Dict:
        if zero_optimization:
            return {"zero_allow_untested_optimizer": zero_allow_untested_optimizer, "zero_optimization": zero_kwargs}
        return {}
