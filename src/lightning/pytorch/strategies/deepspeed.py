# Copyright The Lightning AI team.
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

import logging
from collections import OrderedDict
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.ddp import DDPStrategy

if TYPE_CHECKING:
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


class DeepSpeedStrategy(DDPStrategy):
    strategy_name = "deepspeed"
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: Optional[str] = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 100_000_000,
        max_in_cpu: int = 1_000_000_000,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: Union[str, int] = "auto",
        config: Optional[Union[_PATH, dict[str, Any]]] = None,
        logging_level: int = logging.WARN,
        parallel_devices: Optional[list[torch.device]] = None,
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
        precision_plugin: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        exclude_frozen_parameters: bool = False,
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. *For more information:* :ref:`deepspeed_advanced`.

        .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        *For more information:* https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is compatible with either `precision="16-mixed"` or
                `precision="bf16-mixed"`.

            stage: Different stages of the ZeRO Optimizer. 0 is disabled,
                1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning,
                3 is optimizer+gradient_parameter partitioning using the infinity engine.

            remote_device: Device to instantiate the model on initially (``cpu`` or ``nvme``). Defaults to GPU.

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
                when ``offload_optimizer_device`` is set to ``nvme``.
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
                If set to "auto", the strategy tries to infer this from
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

            exclude_frozen_parameters: Exclude frozen parameters when saving checkpoints.

        """
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
        )
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.deepspeed import (
            DeepSpeedStrategyTrainer as EnterpriseDeepSpeedStrategy,
        )

        self.deepspeed_strategy_impl = EnterpriseDeepSpeedStrategy(
            outer_object=self,
            accelerator=accelerator,
            zero_optimization=zero_optimization,
            stage=stage,
            remote_device=remote_device,
            offload_optimizer=offload_optimizer,
            offload_parameters=offload_parameters,
            offload_params_device=offload_params_device,
            nvme_path=nvme_path,
            params_buffer_count=params_buffer_count,
            params_buffer_size=params_buffer_size,
            max_in_cpu=max_in_cpu,
            offload_optimizer_device=offload_optimizer_device,
            optimizer_buffer_count=optimizer_buffer_count,
            block_size=block_size,
            queue_depth=queue_depth,
            single_submit=single_submit,
            overlap_events=overlap_events,
            thread_count=thread_count,
            pin_memory=pin_memory,
            sub_group_size=sub_group_size,
            contiguous_gradients=contiguous_gradients,
            overlap_comm=overlap_comm,
            allgather_partitions=allgather_partitions,
            reduce_scatter=reduce_scatter,
            allgather_bucket_size=allgather_bucket_size,
            reduce_bucket_size=reduce_bucket_size,
            zero_allow_untested_optimizer=zero_allow_untested_optimizer,
            logging_batch_size_per_gpu=logging_batch_size_per_gpu,
            config=config,
            logging_level=logging_level,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            loss_scale=loss_scale,
            initial_scale_power=initial_scale_power,
            loss_scale_window=loss_scale_window,
            hysteresis=hysteresis,
            min_loss_scale=min_loss_scale,
            partition_activations=partition_activations,
            cpu_checkpointing=cpu_checkpointing,
            contiguous_memory_optimization=contiguous_memory_optimization,
            synchronize_checkpoint_boundary=synchronize_checkpoint_boundary,
            load_full_weights=load_full_weights,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            exclude_frozen_parameters=exclude_frozen_parameters,
        )

    @override
    def setup_environment(self) -> None:
        return self.deepspeed_strategy_impl.setup_environment()

    @override
    def setup_distributed(self) -> None:
        return self.deepspeed_strategy_impl.setup_distributed()

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        return self.deepspeed_strategy_impl.setup(trainer=trainer)

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return self.deepspeed_strategy_impl.restore_checkpoint_after_setup

    @override
    def _setup_model_and_optimizers(
        self, model: Module, optimizers: list[Optimizer]
    ) -> tuple["deepspeed.DeepSpeedEngine", list[Optimizer]]:
        """Setup a model and multiple optimizers together.

        Currently only a single optimizer is supported.

        Return:
            The model wrapped into a :class:`deepspeed.DeepSpeedEngine` and a list with a single
            deepspeed optimizer.

        """
        return self.deepspeed_strategy_impl._setup_model_and_optimizers(model=model, optimizers=optimizers)

    @property
    def zero_stage_3(self) -> bool:
        return self.deepspeed_strategy_impl.zero_stage_3

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        with self.deepspeed_strategy_impl.tensor_init_context(empty_init=empty_init):
            yield

    @contextmanager
    @override
    def model_sharded_context(self) -> Generator[None, None, None]:
        with self.deepspeed_strategy_impl.model_sharded_context():
            yield

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        return self.deepspeed_strategy_impl.distributed_sampler_kwargs

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to

        """
        return self.deepspeed_strategy_impl.setup_optimizers(trainer=trainer)

    @property
    @override
    def handles_gradient_accumulation(self) -> bool:
        """Whether the strategy handles gradient accumulation internally."""
        return self.deepspeed_strategy_impl.handles_gradient_accumulation

    @property
    def deepspeed_engine(self) -> "deepspeed.DeepSpeedEngine":
        return self.deepspeed_strategy_impl.deepspeed_engine

    @property
    def _multi_device(self) -> bool:
        return self.deepspeed_strategy_impl._multi_device

    @override
    def save_checkpoint(self, checkpoint: dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        return self.deepspeed_strategy_impl.save_checkpoint(
            checkpoint=checkpoint, filepath=filepath, storage_options=storage_options
        )

    @override
    def load_checkpoint(self, checkpoint_path: _PATH, weights_only: Optional[bool] = None) -> dict[str, Any]:
        return self.deepspeed_strategy_impl.load_checkpoint(checkpoint_path=checkpoint_path, weights_only=weights_only)

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return self.deepspeed_strategy_impl.lightning_restore_optimizer

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # override to do nothing, deepspeed engine already loaded the weights in `load_checkpoint()`
        return self.deepspeed_strategy_impl.load_model_state_dict(checkpoint=checkpoint, strict=strict)

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        return self.deepspeed_strategy_impl.load_optimizer_state_dict(checkpoint=checkpoint)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("deepspeed", cls, description="Default DeepSpeed Strategy")
        strategy_registry.register("deepspeed_stage_1", cls, description="DeepSpeed with ZeRO Stage 1 enabled", stage=1)
        strategy_registry.register("deepspeed_stage_2", cls, description="DeepSpeed with ZeRO Stage 2 enabled", stage=2)
        strategy_registry.register(
            "deepspeed_stage_2_offload",
            cls,
            description="DeepSpeed ZeRO Stage 2 and CPU Offload",
            stage=2,
            offload_optimizer=True,
        )
        strategy_registry.register("deepspeed_stage_3", cls, description="DeepSpeed ZeRO Stage 3", stage=3)
        strategy_registry.register(
            "deepspeed_stage_3_offload",
            cls,
            description="DeepSpeed ZeRO Stage 3 and CPU Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        )
        strategy_registry.register(
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
    def config(self) -> dict[str, Any]:
        return self.deepspeed_strategy_impl.config

    @property
    def load_full_weights(self) -> bool:
        return self.deepspeed_strategy_impl.load_full_weights
