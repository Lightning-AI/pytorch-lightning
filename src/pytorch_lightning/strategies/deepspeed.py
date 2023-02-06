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
import argparse
import contextlib
import json
import logging
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.utils import _fp_to_half
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn, WarningCache
from pytorch_lightning.utilities.types import LRSchedulerConfig, STEP_OUTPUT

log = logging.getLogger(__name__)
warning_cache = WarningCache()

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
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
        remote_device: str = "cpu",
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
        config: Optional[Union[_PATH, Dict[str, Any]]] = None,
        logging_level: int = logging.WARN,
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
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. `For more information: https://pytorch-
        lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed`.

        .. warning:: ``DeepSpeedStrategy`` is in beta and subject to change.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is compatible with either `precision=16` or
                `precision="bf16"`.

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
                "To use the `DeepSpeedStrategy`, you must have DeepSpeed installed."
                " Install it by running `pip install -U deepspeed`."
            )

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
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
        import deepspeed

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

    def _load_config(self, config: Optional[Union[_PATH, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
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
        assert isinstance(config, dict) or config is None
        return config

    def setup_distributed(self) -> None:
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._init_deepspeed_distributed()
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        _optimizers_to_device(self.optimizers, self.root_device)
        self.init_deepspeed()
        self.barrier()

    def _init_deepspeed_distributed(self) -> None:
        import deepspeed

        assert self.cluster_environment is not None
        if platform.system() != "Windows":
            # do not set env variables on windows, allow deepspeed to control setup
            self._set_node_environment_variables()
            log.info(
                "initializing deepspeed distributed: "
                f"GLOBAL_RANK: {self.global_rank}, "
                f"MEMBER: {self.global_rank + 1}/{self.world_size}"
            )
        self._process_group_backend = self._get_process_group_backend()
        deepspeed.init_distributed(self._process_group_backend, distributed_port=self.cluster_environment.main_port)

    def _set_node_environment_variables(self) -> None:
        assert self.cluster_environment is not None
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    def _setup_model_and_optimizers(
        self, model: Module, optimizers: List[Optimizer]
    ) -> Tuple["deepspeed.DeepSpeedEngine", List[Optimizer]]:
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
        assert self.config is not None
        self.config.setdefault("train_micro_batch_size_per_gpu", 1)
        self.model, optimizer = self._setup_model_and_optimizer(model, optimizers[0])
        self._set_deepspeed_activation_checkpointing()
        return self.model, [optimizer]

    def _setup_model_and_optimizer(
        self,
        model: Module,
        optimizer: Optional[Optimizer],
        lr_scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
    ) -> Tuple["deepspeed.DeepSpeedEngine", Optimizer]:
        """Initialize one model and one optimizer with an optional learning rate scheduler.

        This calls :func:`deepspeed.initialize` internally.
        """
        import deepspeed

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=self.config,
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
        )
        return deepspeed_engine, deepspeed_optimizer

    def init_deepspeed(self) -> None:
        assert self.lightning_module is not None
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

        if not isinstance(self.accelerator, CUDAAccelerator):
            raise MisconfigurationException(
                f"DeepSpeed strategy is only supported on GPU but `{self.accelerator.__class__.__name__}` is used."
            )

        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "DeepSpeed currently does not support different `accumulate_grad_batches` at different epochs."
            )

        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        model = _LightningModuleWrapperBase(forward_module=self.model)

        if self.lightning_module.trainer and self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

    def _init_optimizers(self) -> Tuple[Optimizer, Optional[LRSchedulerConfig], Optional[int]]:
        assert self.lightning_module is not None
        optimizers, lr_schedulers, optimizer_frequencies = _init_optimizers_and_lr_schedulers(self.lightning_module)
        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                "DeepSpeed currently only supports single optimizer, single optional scheduler."
            )
        return (
            optimizers[0],
            lr_schedulers[0] if lr_schedulers else None,
            optimizer_frequencies[0] if optimizer_frequencies else None,
        )

    @property
    def zero_stage_3(self) -> bool:
        assert isinstance(self.config, dict)
        zero_optimization = self.config.get("zero_optimization")
        return zero_optimization is not None and zero_optimization.get("stage") == 3

    def _initialize_deepspeed_train(self, model: Module) -> None:
        optimizer, scheduler = None, None
        assert isinstance(self.config, dict)
        if "optimizer" in self.config:
            rank_zero_info(
                "You have specified an optimizer and/or scheduler within the DeepSpeed config."
                " It is recommended to define it in `LightningModule.configure_optimizers`."
            )
            lr_scheduler = None
        else:
            optimizer, lr_scheduler, _ = self._init_optimizers()
            if lr_scheduler is not None:
                scheduler = lr_scheduler.scheduler

        model, deepspeed_optimizer = self._setup_model_and_optimizer(model, optimizer, scheduler)
        self._set_deepspeed_activation_checkpointing()

        # although we set these here, deepspeed manages the specific optimizer logic
        self.optimizers = [deepspeed_optimizer]

        deepspeed_scheduler = model.lr_scheduler
        if deepspeed_scheduler is not None:
            # disable deepspeed lr scheduling as lightning manages scheduling
            model.lr_scheduler = None
            if lr_scheduler is None:
                lr_scheduler = LRSchedulerConfig(deepspeed_scheduler, interval="step", opt_idx=0)
            else:
                lr_scheduler.scheduler = deepspeed_scheduler
            self.lr_scheduler_configs = [lr_scheduler]
        self.model = model

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        import deepspeed

        if self.zero_stage_3:
            assert self._config_initialized

            if self.precision_plugin.precision == "16":
                dtype = torch.float16
            elif self.precision_plugin.precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            model_parallel_context = deepspeed.zero.Init(
                remote_device=self.remote_device, pin_memory=True, config_dict_or_path=self.config, dtype=dtype
            )
        else:
            model_parallel_context = super().model_sharded_context()

        with model_parallel_context:
            yield

    def _set_deepspeed_activation_checkpointing(self) -> None:
        import deepspeed

        assert isinstance(self.config, dict)
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                profile=checkpoint_config.get("profile"),
            )

    def _initialize_deepspeed_inference(self, model: Module) -> None:
        import deepspeed

        assert isinstance(self.config, dict)

        # todo: this is required for DeepSpeed throughput timers
        inference_config = {"train_micro_batch_size_per_gpu": 1}
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
            optimizer=None,
            lr_scheduler=None,
            model_parameters=[],
            dist_init_required=False,
        )
        self.model = model

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return dict(num_replicas=self.world_size, rank=self.global_rank)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to
        """
        if trainer.state.fn != TrainerFn.FITTING:
            return
        # Skip initializing optimizers here as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_initialize_deepspeed_train`
        # empty optimizers, schedulers and frequencies
        self.optimizers = []
        self.lr_scheduler_configs = []
        self.optimizer_frequencies = []

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return True

    def _format_config(self) -> None:
        if self.config is None:
            raise MisconfigurationException(
                "To use DeepSpeed you must pass in a DeepSpeed config dict, or a path to a JSON config."
                " See: https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed"
            )
        self._format_batch_size_and_grad_accum_config()
        self._format_precision_config()

    def _format_batch_size_and_grad_accum_config(self) -> None:
        # TODO: Using Fabric, we do not support these variables within the config
        assert isinstance(self.config, dict)
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

    def _auto_select_batch_size(self) -> int:
        import deepspeed

        # train_micro_batch_size_per_gpu is used for throughput logging purposes
        # by default we try to use the batch size of the loader
        assert self.lightning_module is not None
        batch_size = 1
        train_dl_source = self.lightning_module.trainer._data_connector._train_dataloader_source
        if train_dl_source.is_defined():
            try:
                train_dataloader = train_dl_source.dataloader()
                if hasattr(train_dataloader, "batch_sampler"):
                    batch_size = train_dataloader.batch_sampler.batch_size  # type: ignore[union-attr]
            # broad exception on purpose as `source.dataloader()` will fail if the dataloader requires `setup`
            # to have been called before
            except Exception:
                if self.global_rank == 0:
                    deepspeed.utils.logging.logger.warning(
                        "Tried to infer the batch size for internal deepspeed logging from the `train_dataloader()`. "
                        "To ensure DeepSpeed logging remains correct, please manually pass the plugin with the "
                        "batch size, `Trainer(strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=batch_size))`."
                    )
        return batch_size

    def _format_precision_config(self) -> None:
        assert isinstance(self.config, dict)
        if self.precision_plugin.precision == "16":
            if "fp16" not in self.config and self.precision_plugin.amp_type == "native":
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
            elif "amp" not in self.config and self.precision_plugin.amp_type == "apex":
                rank_zero_info("Enabling DeepSpeed APEX Implementation.")
                self.config["amp"] = {"enabled": True, "opt_level": self.precision_plugin.amp_level}
        elif "bf16" not in self.config and self.precision_plugin.precision == "bf16":
            rank_zero_info("Enabling DeepSpeed BF16.")
            self.config["bf16"] = {"enabled": True}

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
        **zero_kwargs: Any,
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
    def deepspeed_engine(self) -> "deepspeed.DeepSpeedEngine":
        return self.model

    @property
    def _multi_device(self) -> bool:
        return self.num_processes > 1 or self.num_nodes > 1

    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        # broadcast the filepath from rank 0 to ensure all the states are saved in a common filepath
        filepath = self.broadcast(filepath)

        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used."
            )

        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            warning_cache.warn(
                "When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint")

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        if self.load_full_weights and self.zero_stage_3:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.broadcast(checkpoint_path)
            return super().load_checkpoint(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        assert self.lightning_module is not None

        from pytorch_lightning.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        _, client_state = self.deepspeed_engine.load_checkpoint(
            checkpoint_path, load_optimizer_states=is_fitting, load_lr_scheduler_states=False
        )
        if client_state is None:
            raise MisconfigurationException(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint "
                "or a single checkpoint file with `Trainer(strategy=DeepSpeedStrategy(load_full_weights=True))`."
            )
        return client_state

    @property
    def lightning_restore_optimizer(self) -> bool:
        assert self.lightning_module is not None
        # managed by DeepSpeed
        if self.load_full_weights and self.zero_stage_3 and self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            rank_zero_warn(
                "A single checkpoint file has been given. This means optimizer states cannot be restored."
                " If you'd like to restore these states, you must provide a path to the originally saved DeepSpeed"
                " checkpoint. When using ZeRO 3, the original path should be a directory."
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
        import deepspeed

        assert self.lightning_module is not None

        def load(module: torch.nn.Module, prefix: str = "") -> None:

            missing_keys: List[str] = []
            unexpected_keys: List[str] = []
            error_msgs: List[str] = []
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
    def register_strategies(cls, strategy_registry: Dict) -> None:
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

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        batch = apply_to_collection(batch, Tensor, function=_fp_to_half, precision=self.precision_plugin.precision)
        return super().batch_to_device(batch, device, dataloader_idx)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)
