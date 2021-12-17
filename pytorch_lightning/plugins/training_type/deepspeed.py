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
import argparse
import contextlib
import json
import logging
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import AMPType, GradClipAlgorithmType
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import log, rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import _PATH, LRSchedulerTypeTuple
from pytorch_lightning.utilities.warnings import rank_zero_warn, WarningCache

warning_cache = WarningCache()

if _DEEPSPEED_AVAILABLE:
    import deepspeed


def remove_module_hooks(model: torch.nn.Module) -> None:
    # todo (tchaton) awaiting this feature to move upstream to DeepSpeed
    for module in model.modules():
        module._backward_hooks = OrderedDict()
        module._is_full_backward_hook = None
        module._forward_hooks = OrderedDict()
        module._forward_pre_hooks = OrderedDict()
        module._state_dict_hooks = OrderedDict()
        module._load_state_dict_pre_hooks = OrderedDict()


class LightningDeepSpeedModule(_LightningModuleWrapperBase):
    def __init__(self, pl_module: "pl.LightningModule", precision: int) -> None:
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
    distributed_backend = DistributedType.DEEPSPEED
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str = "cpu",
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 1e8,
        max_in_cpu: int = 1e9,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1e12,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 2e8,
        reduce_bucket_size: int = 2e8,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: Union[str, int] = "auto",
        config: Optional[Union[Path, str, dict]] = None,
        logging_level: int = logging.WARN,
        num_nodes: Optional[int] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. `For more information: https://pytorch-
        lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed`.

        .. warning:: ``DeepSpeedPlugin`` is in beta and subject to change.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is only compatible with precision=16.

            stage: Different stages of the ZeRO Optimizer. 0 is disabled,
                1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning,
                3 is optimizer+gradient_parameter partitioning using the infinity engine.

            remote_device: Device to instantiate the model on initially (``cpu`` or ``nvme``).

            offload_optimizer: Enable offloading optimizer memory and computation to CPU or NVMe
                based on ``offload_optimizer_device``.

            offload_parameters: When using ZeRO Stage 3, Enable offloading parameter memory and computation
                to CPU or NVMe based on ``offload_params_device``.

            offload_params_device: When offloading parameters choose the device to offload to, ``cpu`` or ``nvme``.

            offload_optimizer_device: When offloading optimizer state choose the device to offload to,
                ``cpu`` or ``nvme``.

            params_buffer_count: Number of buffers in buffer pool for
                parameter offloading when ``offload_params_device`` is ``nvme``.

            params_buffer_size: Size of buffers in buffer pool for parameter offloading
                when ``offload_params_device`` is ``nvme``.

            max_in_cpu: Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.

            nvme_path: Filesystem path for NVMe device for optimizer/parameter state offloading.

            optimizer_buffer_count: Number of buffers in buffer pool for optimizer state offloading
                when ``offload_optimizer_device`` is set to to ``nvme``.
                This should be at least the number of states maintained per parameter by the optimizer.
                For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance).

            block_size: When using NVMe Offloading, the I/O block size in bytes.

            queue_depth: When using NVMe Offloading, the I/O queue depth.

            single_submit: When using NVMe Offloading,
                submit requests to storage device as multiple individual requests,
                as opposed to one block of requests.

            overlap_events: When using NVMe Offloading,
                submit requests to storage device in an overlapped fashion
                without waiting for completion of earlier requests.

            thread_count: When using NVMe Offloading,
                Intra-request parallelism for each read/write submitted by a user thread.

            pin_memory: When using ZeRO stage 3, pin optimizer state memory on CPU.
                This could boost throughput at the cost of extra memory overhead.

            sub_group_size: When using ZeRO stage 3, defines the number of parameters
                within a sub group to offload at a time.
                Smaller numbers require more communication, but improve memory efficiency.

            contiguous_gradients: Copies gradients to a continuous buffer as they are produced.
                Avoids memory fragmentation during backwards. Useful when training large models.

            overlap_comm: Overlap the reduction (synchronization) of gradients with the backwards computation.
                This is a speed optimization when training across multiple GPUs/machines.

            allgather_partitions: All gather updated parameters at the end of training step,
                instead of using a series of broadcast collectives.

            reduce_scatter: Use reduce/scatter instead of allreduce to average gradients.

            allgather_bucket_size: Number of elements to allgather at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            reduce_bucket_size: Number of elements to reduce at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            zero_allow_untested_optimizer: Allow untested optimizers to be used with ZeRO. Currently only Adam is a
                DeepSpeed supported optimizer when using ZeRO.

            logging_batch_size_per_gpu: Config used in DeepSpeed to calculate verbose timing for logging
                on a per sample per second basis (only displayed if logging=logging.INFO).
                If set to "auto", the plugin tries to infer this from
                the train DataLoader's BatchSampler, else defaults to 1.
                To obtain accurate logs when using datasets that do not support batch samplers,
                set this to the actual per gpu batch size (trainer.batch_size).

            config: Pass in a deepspeed formatted config dict,
                or path to a deepspeed config: https://www.deepspeed.ai/docs/config-json.
                All defaults will be ignored if a config is passed in.

            logging_level: Set logging level for deepspeed.

            loss_scale: Loss scaling value for FP16 training.
                0.0 results in dynamic loss scaling, otherwise static.

            initial_scale_power: Power of the initial dynamic loss scale value. Loss scale is computed
                by ``2^initial_scale_power``.

            loss_scale_window: Window in which to raise/lower the dynamic FP16 loss scaling value.

            hysteresis: FP16 Delay shift in Dynamic Loss scaling.

            min_loss_scale: The minimum FP16 dynamic loss scaling value.

            partition_activations: Enables partition activation when used with ZeRO stage 3 and model parallelism.
                Still requires you to wrap your forward functions in deepspeed.checkpointing.checkpoint.
                See `deepspeed tutorial
                <https://www.deepspeed.ai/tutorials/megatron/#deepspeed-activation-checkpoints-optional>`_.

            cpu_checkpointing: Offloads partitioned activations to CPU if ``partition_activations`` is enabled.

            contiguous_memory_optimization: Copies partitioned activations so that they are contiguous in memory.
                Not supported by all models.

            synchronize_checkpoint_boundary: Insert :func:`torch.cuda.synchronize` at each checkpoint boundary.

            load_full_weights: True when loading a single checkpoint file containing the model state dict
                when using ZeRO Stage 3. This differs from the DeepSpeed checkpoint which contains shards
                per worker.
        """
        if not _DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the DeepSpeed plugin, you must have DeepSpeed installed. pip install deepspeed"
            )

        super().__init__(
            parallel_devices=parallel_devices,
            num_nodes=num_nodes,
            cluster_environment=cluster_environment,
        )

        self.config = self._load_config(config)
        if self.config is None:
            # User has not overridden config, set defaults
            self.config = self._create_default_config(
                zero_optimization,
                zero_allow_untested_optimizer,
                logging_batch_size_per_gpu,
                offload_optimizer=offload_optimizer,
                offload_parameters=offload_parameters,
                nvme_path=nvme_path,
                offload_params_device=offload_params_device,
                params_buffer_count=params_buffer_count,
                params_buffer_size=params_buffer_size,
                max_in_cpu=max_in_cpu,
                pin_memory=pin_memory,
                offload_optimizer_device=offload_optimizer_device,
                optimizer_buffer_count=optimizer_buffer_count,
                block_size=block_size,
                queue_depth=queue_depth,
                single_submit=single_submit,
                overlap_events=overlap_events,
                thread_count=thread_count,
                partition_activations=partition_activations,
                cpu_checkpointing=cpu_checkpointing,
                contiguous_memory_optimization=contiguous_memory_optimization,
                synchronize_checkpoint_boundary=synchronize_checkpoint_boundary,
                stage=stage,
                contiguous_gradients=contiguous_gradients,
                overlap_comm=overlap_comm,
                allgather_partitions=allgather_partitions,
                reduce_scatter=reduce_scatter,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size,
                sub_group_size=sub_group_size,
            )
        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

        self.remote_device = remote_device
        self.load_full_weights = load_full_weights

        # default FP16 parameters.
        self.loss_scale = loss_scale
        self.initial_scale_power = initial_scale_power
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale

        # optionally set by Lite
        self._precision: Optional[Union[str, int]] = None
        self._amp_level: Optional[str] = None
        self._amp_type: Optional[str] = None

    @property
    def precision(self) -> Union[str, int]:
        return self._precision or self.lightning_module.trainer.precision

    @property
    def amp_level(self) -> Optional[str]:
        if self._amp_type == AMPType.APEX:
            return self._amp_level or self.lightning_module.trainer._accelerator_connector.amp_level

    @property
    def amp_type(self) -> Optional[str]:
        return self._amp_type or self.lightning_module.trainer._accelerator_connector.amp_type

    def _load_config(self, config):
        if config is None and self.DEEPSPEED_ENV_VAR in os.environ:
            rank_zero_info(f"Loading DeepSpeed config from set {self.DEEPSPEED_ENV_VAR} environment variable")
            config = os.environ[self.DEEPSPEED_ENV_VAR]
        if isinstance(config, (str, Path)):
            if not os.path.isfile(config):
                raise MisconfigurationException(
                    f"You passed in a path to a DeepSpeed config but the path does not exist: {config}"
                )
            with open(config) as f:
                config = json.load(f)
        return config

    def setup_distributed(self):
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._init_deepspeed_distributed()

        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

    def _init_deepspeed_distributed(self) -> None:
        if platform.system() != "Windows":
            # do not set env variables on windows, allow deepspeed to control setup
            self._set_node_environment_variables()
            log.info(
                "initializing deepspeed distributed: "
                f"GLOBAL_RANK: {self.global_rank}, "
                f"MEMBER: {self.global_rank + 1}/{self.world_size}"
            )
        deepspeed.init_distributed(
            self.torch_distributed_backend, distributed_port=self.cluster_environment.master_port()
        )

    def _set_node_environment_variables(self) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    @property
    def restore_checkpoint_after_pre_dispatch(self) -> bool:
        return True

    def pre_dispatch(self):
        self.init_deepspeed()
        self.barrier()

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Setup a model and multiple optimizers together.

        Currently only a single optimizer is supported.

        Return:
            The model wrapped into a :class:`deepspeed.DeepSpeedEngine` and a list with a single
            deepspeed optimizer.
        """
        if len(optimizers) != 1:
            raise ValueError(
                f"Currently only one optimizer is supported with DeepSpeed."
                f" Got {len(optimizers)} optimizers instead."
            )

        # train_micro_batch_size_per_gpu is used for throughput logging purposes
        # normally we set this to the batch size, but it is not available here unless the user provides it
        # as part of the config
        self.config.setdefault("train_micro_batch_size_per_gpu", 1)
        self._model, optimizer = self._setup_model_and_optimizer(model, optimizers[0])
        self._set_deepspeed_activation_checkpointing()
        return self._model, [optimizer]

    def _setup_model_and_optimizer(
        self, model: Module, optimizer: Optimizer, lr_scheduler: Optional[_LRScheduler] = None
    ):
        """Initialize one model and one optimizer with an optional learning rate scheduler.

        This calls :func:`deepspeed.initialize` internally.
        """
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=self.config,
            model=model,
            model_parameters=model_parameters,  # type: ignore
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
        )
        return deepspeed_engine, deepspeed_optimizer

    def init_deepspeed(self):
        # deepspeed handles gradient clipping internally
        if is_overridden("configure_gradient_clipping", self.lightning_module, pl.LightningModule):
            rank_zero_warn(
                "Since DeepSpeed handles gradient clipping internally, the default"
                " `LightningModule.configure_gradient_clipping` implementation will not actually clip gradients."
                " The hook will still be called. Consider setting"
                " `Trainer(gradient_clip_val=..., gradient_clip_algorithm='norm')`"
                " which will use the internal mechanism."
            )

        if self.lightning_module.trainer.gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            raise MisconfigurationException("DeepSpeed does not support clipping gradients by value.")

        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "DeepSpeed currently does not support different `accumulate_grad_batches` at different epochs."
            )

        precision = self.lightning_module.trainer.accelerator.precision
        model = LightningDeepSpeedModule(pl_module=self.model, precision=precision)

        if self.lightning_module.trainer and self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

    def _init_optimizers(self) -> Tuple[Optimizer, Optional[Union[LRSchedulerTypeTuple]], Optional[int]]:
        optimizers, schedulers, optimizer_frequencies = self.lightning_module.trainer.init_optimizers(
            self.lightning_module
        )
        if len(optimizers) > 1 or len(schedulers) > 1:
            raise MisconfigurationException(
                "DeepSpeed currently only supports single optimizer, single optional scheduler."
            )
        return (
            optimizers[0],
            schedulers[0] if schedulers else _get_default_scheduler_config(),
            optimizer_frequencies[0] if optimizer_frequencies else None,
        )

    @property
    def zero_stage_3(self) -> bool:
        return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3

    def _initialize_deepspeed_train(self, model):
        if "optimizer" in self.config:
            optimizer, lr_scheduler = None, _get_default_scheduler_config()
        else:
            rank_zero_info(
                "You have not specified an optimizer or scheduler within the DeepSpeed config."
                " Using `configure_optimizers` to define optimizer and scheduler."
            )
            optimizer, lr_scheduler, _ = self._init_optimizers()

        scheduler = lr_scheduler["scheduler"]
        model, deepspeed_optimizer = self._setup_model_and_optimizer(model, optimizer, scheduler)
        self._set_deepspeed_activation_checkpointing()

        # although we set these here, deepspeed manages the specific optimizer logic
        self.lightning_module.trainer.optimizers = [deepspeed_optimizer]

        deepspeed_scheduler = model.lr_scheduler
        if deepspeed_scheduler is not None:
            # disable deepspeed lr scheduling as lightning manages scheduling
            model.lr_scheduler = None
            lr_scheduler["scheduler"] = deepspeed_scheduler
            self.lightning_module.trainer.lr_schedulers = [lr_scheduler]
        self.model = model

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        if self.zero_stage_3:
            assert self._config_initialized
            dtype = torch.float16 if self.precision in (16, "mixed") else torch.float32
            model_parallel_context = deepspeed.zero.Init(
                remote_device=self.remote_device, pin_memory=True, config_dict_or_path=self.config, dtype=dtype
            )
        else:
            model_parallel_context = super().model_sharded_context()

        with model_parallel_context:
            yield

    def _set_deepspeed_activation_checkpointing(self):
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                profile=checkpoint_config.get("profile"),
            )

    def _initialize_deepspeed_inference(self, model):
        # todo: Currently DeepSpeed requires optimizers at inference to partition weights correctly
        optimizer, scheduler = None, None
        if "optimizer" not in self.config:
            rank_zero_info(
                "You have not specified an optimizer or scheduler within the DeepSpeed config."
                " Using `configure_optimizers` to define optimizer and scheduler."
            )
            optimizer, lr_scheduler, _ = self._init_optimizers()
            scheduler = lr_scheduler["scheduler"]
        inference_config = {
            # todo: this is required for DeepSpeed throughput timers
            "train_micro_batch_size_per_gpu": 1
        }
        if "fp16" in self.config:
            inference_config.update({"fp16": self.config["fp16"]})
        if self.zero_stage_3:
            inference_config.update(
                {
                    "zero_allow_untested_optimizer": self.config["zero_allow_untested_optimizer"],
                    "zero_optimization": self.config["zero_optimization"],
                }
            )
        # Remove all module hooks before initializing new model
        remove_module_hooks(model)
        model, _, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=inference_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=[],
            dist_init_required=False,
        )
        self.model = model

    @property
    def lightning_module(self):
        # the model may not be wrapped with DeepEngine & LightningDeepSpeedModule if calling this too early
        module = getattr(self.model, "module", self.model)
        return module.module if isinstance(module, LightningDeepSpeedModule) else module

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=self.world_size, rank=self.global_rank)
        return distributed_sampler_kwargs

    def init_optimizers(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> Tuple[List, List, List]:
        # Skip initializing optimizers here as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_initialize_deepspeed_train`
        return [], [], []  # empty optimizers, schedulers and frequencies

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return True

    def _format_config(self):
        if self.config is None:
            raise MisconfigurationException(
                "To use DeepSpeed you must pass in a DeepSpeed config dict, or a path to a JSON config."
                " See: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed"
            )
        self._format_batch_size_and_grad_accum_config()
        self._format_precision_config()

    def _format_batch_size_and_grad_accum_config(self):
        # todo: using lite, we do not support these variables within the config
        if self.lightning_module is None:
            return

        if "gradient_accumulation_steps" in self.config:
            raise MisconfigurationException(
                "Do not set `gradient_accumulation_steps` in the DeepSpeed config"
                " as this will be set with the `accumulate_grad_batches` argument passed via the Lightning Trainer."
            )
        self.config["gradient_accumulation_steps"] = self.lightning_module.trainer.accumulate_grad_batches
        if "train_micro_batch_size_per_gpu" not in self.config:
            batch_size = self._auto_select_batch_size()
            self.config["train_micro_batch_size_per_gpu"] = batch_size
        if "gradient_clipping" not in self.config:
            self.config["gradient_clipping"] = self.lightning_module.trainer.gradient_clip_val or 0.0

    def _auto_select_batch_size(self):
        # train_micro_batch_size_per_gpu is used for throughput logging purposes
        # by default we try to use the batch size of the loader
        batch_size = 1
        train_dl_source = self.lightning_module.trainer._data_connector._train_dataloader_source
        if train_dl_source.is_defined():
            try:
                train_dataloader = train_dl_source.dataloader()
                if hasattr(train_dataloader, "batch_sampler"):
                    batch_size = train_dataloader.batch_sampler.batch_size
            # broad exception on purpose as `source.dataloader()` will fail if the dataloader requires `setup`
            # to have been called before
            except Exception:
                if self.global_rank == 0:
                    deepspeed.utils.logging.logger.warning(
                        "Tried to infer the batch size for internal deepspeed logging from the `train_dataloader()`. "
                        "To ensure DeepSpeed logging remains correct, please manually pass the plugin with the "
                        "batch size, `Trainer(strategy=DeepSpeedPlugin(logging_batch_size_per_gpu=batch_size))`."
                    )
        return batch_size

    def _format_precision_config(self):
        if self.amp_type == AMPType.APEX:
            amp_level = self.amp_level
        if self.precision in (16, "mixed"):
            if "fp16" not in self.config and self.amp_type == AMPType.NATIVE:
                # FP16 is a DeepSpeed standalone AMP implementation
                rank_zero_info("Enabling DeepSpeed FP16.")
                self.config["fp16"] = {
                    "enabled": True,
                    "loss_scale": self.loss_scale,
                    "initial_scale_power": self.initial_scale_power,
                    "loss_scale_window": self.loss_scale_window,
                    "hysteresis": self.hysteresis,
                    "min_loss_scale": self.min_loss_scale,
                }
            elif "amp" not in self.config and self.amp_type == AMPType.APEX:
                rank_zero_only("Enabling DeepSpeed APEX Implementation.")
                self.config["amp"] = {"enabled": True, "opt_level": amp_level}

    def _create_default_config(
        self,
        zero_optimization: bool,
        zero_allow_untested_optimizer: bool,
        logging_batch_size_per_gpu: Union[str, int],
        partition_activations: bool,
        cpu_checkpointing: bool,
        contiguous_memory_optimization: bool,
        synchronize_checkpoint_boundary: bool,
        offload_optimizer: bool,
        offload_parameters: bool,
        nvme_path: str,
        offload_params_device: str,
        params_buffer_count: int,
        params_buffer_size: int,
        max_in_cpu: int,
        offload_optimizer_device: str,
        optimizer_buffer_count: int,
        pin_memory: bool,
        block_size: int,
        queue_depth: int,
        single_submit: bool,
        overlap_events: bool,
        thread_count: int,
        **zero_kwargs,
    ) -> Dict:
        cfg = {
            "activation_checkpointing": {
                "partition_activations": partition_activations,
                "cpu_checkpointing": cpu_checkpointing,
                "contiguous_memory_optimization": contiguous_memory_optimization,
                "synchronize_checkpoint_boundary": synchronize_checkpoint_boundary,
            },
            "aio": {
                "block_size": block_size,
                "queue_depth": queue_depth,
                "single_submit": single_submit,
                "overlap_events": overlap_events,
                "thread_count": thread_count,
            },
        }
        if zero_optimization:
            zero_config = zero_kwargs

            if offload_optimizer:
                zero_config["offload_optimizer"] = {
                    "device": offload_optimizer_device,
                    "nvme_path": nvme_path,
                    "buffer_count": optimizer_buffer_count,
                    "pin_memory": pin_memory,
                }
            if offload_parameters:
                zero_config["offload_param"] = {
                    "device": offload_params_device,
                    "nvme_path": nvme_path,
                    "buffer_count": params_buffer_count,
                    "buffer_size": params_buffer_size,
                    "max_in_cpu": max_in_cpu,
                    "pin_memory": pin_memory,
                }
            cfg = {
                "zero_allow_untested_optimizer": zero_allow_untested_optimizer,
                "zero_optimization": zero_config,
                **cfg,
            }
        if logging_batch_size_per_gpu != "auto":
            cfg = {"train_micro_batch_size_per_gpu": logging_batch_size_per_gpu, **cfg}
        return cfg

    @property
    def deepspeed_engine(self):
        return self.model

    @property
    def _multi_device(self) -> bool:
        return self.num_processes > 1 or self.num_nodes > 1

    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
        """
        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            warning_cache.warn(
                "When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states", "lr_schedulers"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint)

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        if self.load_full_weights and self.zero_stage_3:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.broadcast(checkpoint_path)
            return super().load_checkpoint(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        from pytorch_lightning.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        _, client_state = self.deepspeed_engine.load_checkpoint(
            checkpoint_path, load_optimizer_states=is_fitting, load_lr_scheduler_states=is_fitting
        )
        if client_state is None:
            raise MisconfigurationException(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint "
                "or a single checkpoint file with `Trainer(strategy=DeepSpeedPlugin(load_full_weights=True))`."
            )
        return client_state

    @property
    def lightning_restore_optimizer_and_schedulers(self) -> bool:
        # managed by DeepSpeed
        if self.load_full_weights and self.zero_stage_3 and self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            rank_zero_warn(
                "A single checkpoint file has been given. This means optimizer states and "
                "scheduler states can not be restored. If you'd like to restore these states, you must "
                "provide a path to the originally saved DeepSpeed checkpoint."
            )
        return False

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the weights in `load_checkpoint()`
        if self.load_full_weights and self.zero_stage_3:
            self.model_to_device()
            self._restore_zero_state(checkpoint)

    def _restore_zero_state(self, ckpt: Mapping[str, Any]) -> None:
        """Overrides the normal load_state_dict behaviour in PyTorch to ensure we gather parameters that may be
        sharded across processes before loading the state dictionary when using ZeRO stage 3. This is then
        automatically synced across processes.

        Args:
            ckpt: The ckpt file.
        """

        def load(module: torch.nn.Module, prefix=""):

            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            state_dict = ckpt["state_dict"]

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if self.is_global_zero:
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=True,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self.lightning_module, prefix="")

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the states in `load_checkpoint()`
        pass

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("deepspeed", cls, description="Default DeepSpeed Plugin")
        plugin_registry.register("deepspeed_stage_1", cls, description="DeepSpeed with ZeRO Stage 1 enabled", stage=1)
        plugin_registry.register("deepspeed_stage_2", cls, description="DeepSpeed with ZeRO Stage 2 enabled", stage=2)
        plugin_registry.register(
            "deepspeed_stage_2_offload",
            cls,
            description="DeepSpeed ZeRO Stage 2 and CPU Offload",
            stage=2,
            offload_optimizer=True,
        )
        plugin_registry.register("deepspeed_stage_3", cls, description="DeepSpeed ZeRO Stage 3", stage=3)
        plugin_registry.register(
            "deepspeed_stage_3_offload",
            cls,
            description="DeepSpeed ZeRO Stage 3 and CPU Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        )
        plugin_registry.register(
            "deepspeed_stage_3_offload_nvme",
            cls,
            description="DeepSpeed ZeRO Stage 3 and NVMe Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="nvme",
            offload_params_device="nvme",
            offload_optimizer_device="nvme",
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, plugin: CheckpointIO) -> None:
        raise MisconfigurationException("DeepSpeed currently does not support custom checkpoint plugins.")

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)
