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
import contextlib
import json
import logging
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, Union

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import log, rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.types import LRSchedulerTypeTuple
from pytorch_lightning.utilities.warnings import _warn, LightningDeprecationWarning

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
    distributed_backend = "deepspeed"
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
        save_full_weights: bool = True,
        cpu_offload: bool = False,
        cpu_offload_params: bool = False,
        cpu_offload_use_pin_memory: bool = False,
    ) -> None:
        """
        Provides capabilities to run training using the DeepSpeed library,
        with training optimizations for large billion parameter models.
        `For more information: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed`.

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

            save_full_weights: Gathers weights across all processes before saving to disk
                when using ZeRO Stage 3. This allows a single weight file to contain the entire model,
                rather than individual sharded weight files.
                Disable to save sharded states individually.
        """
        if not _DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the DeepSpeed plugin, you must have DeepSpeed installed. pip install deepspeed"
            )

        if cpu_offload or cpu_offload_params or cpu_offload_use_pin_memory:
            _warn(
                "The usage of `cpu_offload`, `cpu_offload_params`, and `cpu_offload_use_pin_memory` "
                "is deprecated since v1.4 and will be removed in v1.5."
                " From now on use `offload_optimizer`, `offload_parameters` and `pin_memory`.",
                category=LightningDeprecationWarning,
            )
            offload_optimizer = cpu_offload
            offload_parameters = cpu_offload_params
            pin_memory = cpu_offload_use_pin_memory

        super().__init__(
            parallel_devices=parallel_devices, num_nodes=num_nodes, cluster_environment=cluster_environment
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
        self.save_full_weights = save_full_weights

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
        if isinstance(config, (str, Path)):
            if not os.path.isfile(config):
                raise MisconfigurationException(
                    f"You passed in a path to a DeepSpeed config but the path does not exist: {config}"
                )
            with open(config) as f:
                config = json.load(f)
        return config

    def setup_distributed(self):
        super().setup_distributed()
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

    def init_ddp_connection(self, global_rank: Optional[int] = None, world_size: Optional[int] = None) -> None:
        if platform.system() != "Windows":
            # do not set env variables on windows, allow deepspeed to control setup
            global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
            world_size = world_size if world_size is not None else self.cluster_environment.world_size()
            self._set_node_environment_variables(global_rank, world_size)
            log.info(
                f"initializing deepspeed distributed: "
                f"GLOBAL_RANK: {global_rank}, "
                f"MEMBER: {global_rank + 1}/{world_size}"
            )
        deepspeed.init_distributed(
            self.torch_distributed_backend, distributed_port=self.cluster_environment.master_port()
        )

    def _set_node_environment_variables(
        self, global_rank: Optional[int] = None, world_size: Optional[int] = None
    ) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def pre_dispatch(self):
        self.init_deepspeed()
        self.barrier()

    def init_deepspeed(self):
        self._handle_gradient_accumulation_steps()

        precision = self.lightning_module.trainer.accelerator.precision
        model = LightningDeepSpeedModule(pl_module=self.model, precision=precision)

        if self.zero_stage_3:
            # Ensure the entire model has been moved to the appropriate device
            dtype = torch.float16 if self.precision in (16, "mixed") else torch.float32
            deepspeed.zero.Init(
                module=model, remote_device=self.remote_device, pin_memory=True, config=self.config, dtype=dtype
            )

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
                "Using `configure_optimizers` to define optimizer and scheduler."
            )
            optimizer, lr_scheduler, _ = self._init_optimizers()

        scheduler = lr_scheduler["scheduler"]

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model, deepspeed_optimizer, _, deepspeed_scheduler = deepspeed.initialize(
            config=self.config,
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            dist_init_required=False,
        )

        self._set_deepspeed_activation_checkpointing()

        # although we set these here, deepspeed manages the specific optimizer logic
        self.lightning_module.trainer.optimizers = [deepspeed_optimizer]
        if deepspeed_scheduler is not None:
            lr_scheduler["scheduler"] = deepspeed_scheduler
            self.lightning_module.trainer.lr_schedulers = [lr_scheduler]
        self.model = model

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        if self.zero_stage_3:
            assert self._config_initialized
            dtype = torch.float16 if self.precision in (16, "mixed") else torch.float32
            model_parallel_context = deepspeed.zero.Init(
                remote_device=self.remote_device, pin_memory=True, config=self.config, dtype=dtype
            )
        else:
            model_parallel_context = super().model_sharded_context()

        with model_parallel_context:
            yield

    @property
    def precision(self) -> Union[str, int]:
        return self.lightning_module.trainer.precision

    def _set_deepspeed_activation_checkpointing(self):
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_checkpointing"),
                checkpoint_in_cpu=checkpoint_config.get("checkpoint_in_cpu"),
                profile=checkpoint_config.get("profile"),
            )

    def _initialize_deepspeed_inference(self, model):
        # todo: Currently DeepSpeed requires optimizers at inference to partition weights correctly
        optimizer, scheduler = None, None
        if "optimizer" not in self.config:
            rank_zero_info(
                "You have not specified an optimizer or scheduler within the DeepSpeed config."
                "Using `configure_optimizers` to define optimizer and scheduler."
            )
            optimizer, lr_scheduler, _ = self._init_optimizers()
            scheduler = lr_scheduler["scheduler"]
        inference_config = {
            # todo: this is required for DeepSpeed throughput timers, or throughput timers will be incorrect
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

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        # note: We rely on the deepspeed engine to carry out the step rather than the optimizer.
        # internally, the engine has a reference to the optimizer already.
        self.model.step(**kwargs)

    def _handle_gradient_accumulation_steps(self):
        """
        This functions overrides the trainer.accumulation_scheduler to generate
        ``accumulate_grad_batches=1``.
        Therefore, ``optimizer_step`` will be called on every batches seen
        so DeepSpeed Engine handles the gradient accumulation logic internally.
        """
        if self.config.get("gradient_accumulation_steps") > 1:
            self._original_accumulate_grad_batches = self.lightning_module.trainer.accumulate_grad_batches
            # todo (tchaton) Add support for accumulate_grad_batches being a dictionary.
            self.lightning_module.trainer.accumulation_scheduler = GradientAccumulationScheduler({0: 1})
        else:
            self._original_accumulate_grad_batches = None

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
            batch_size = self._auto_select_batch_size()
            self.config["train_micro_batch_size_per_gpu"] = batch_size
        self.config["gradient_accumulation_steps"] = self.lightning_module.trainer.accumulate_grad_batches
        if "gradient_clipping" not in self.config:
            self.config["gradient_clipping"] = self.lightning_module.trainer.gradient_clip_val

    def _auto_select_batch_size(self):
        # train_micro_batch_size_per_gpu is used for throughput logging purposes
        # by default we try to use the batch size of the loader
        batch_size = 1
        if hasattr(self.lightning_module, "train_dataloader"):
            train_dataloader = self.lightning_module.train_dataloader()
            if hasattr(train_dataloader, "batch_sampler"):
                batch_size = train_dataloader.batch_sampler.batch_size
        return batch_size

    def _format_precision_config(self):
        amp_type = self.lightning_module.trainer.accelerator_connector.amp_type
        amp_level = self.lightning_module.trainer.accelerator_connector.amp_level
        precision = self.lightning_module.trainer.accelerator_connector.precision
        if precision in (16, "mixed"):
            if "fp16" not in self.config and amp_type == AMPType.NATIVE:
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
            elif "amp" not in self.config and amp_type == AMPType.APEX:
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

    def _filepath_to_dir(self, filepath: str) -> str:
        return os.path.dirname(filepath)

    @property
    def deepspeed_engine(self):
        return self.model

    def save_checkpoint(self, checkpoint: Dict, filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
        """
        if self.world_size > 1 and self.zero_stage_3:
            if self.save_full_weights:
                # todo: expose this as general function in deepspeed
                state_dict = self.deepspeed_engine._zero3_consolidated_fp16_state_dict()
                if self.is_global_zero:
                    # State dict keys will include reference to wrapper LightningDeepSpeedModule
                    # Delete `module` prefix before saving.
                    state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
                    checkpoint["state_dict"] = state_dict
                    return super().save_checkpoint(checkpoint, filepath)
                return

            # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
            # dump states as a checkpoint dictionary object
            save_dir = self._filepath_to_dir(filepath)
            _exclude_keys = ["state_dict", "optimizer_states", "lr_schedulers"]
            checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
            self.deepspeed_engine.save_checkpoint(save_dir, client_state=checkpoint)
        else:
            super().save_checkpoint(checkpoint, filepath)

    def load_checkpoint_file(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        if self.save_full_weights or self.world_size == 1:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.broadcast(checkpoint_path)
            return super().load_checkpoint_file(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        from pytorch_lightning.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        save_dir = self._filepath_to_dir(checkpoint_path)

        if self.zero_stage_3:
            # TODO: Currently required as this call is missing within the deepspeed engine.
            self.deepspeed_engine.optimizer._partition_all_parameters()

        _, client_state = self.deepspeed_engine.load_checkpoint(
            save_dir, load_optimizer_states=is_fitting, load_lr_scheduler_states=is_fitting
        )
        return client_state

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the weights in `load_checkpoint_file()`
        pass

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the states in `load_checkpoint_file()`
        pass

    def update_global_step(self, total_batch_idx: int, current_global_step: int) -> int:
        if self._original_accumulate_grad_batches is None:
            return super().update_global_step(total_batch_idx, current_global_step)
        if total_batch_idx % self._original_accumulate_grad_batches == 0:
            current_global_step += 1
        return current_global_step

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("deepspeed", cls, description="Default DeepSpeed Plugin")
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
