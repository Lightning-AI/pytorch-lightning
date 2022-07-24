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
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.accelerators.hpu import HPUAccelerator
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import log
from pytorch_lightning.utilities.enums import AMPType, PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import _PATH, LRSchedulerConfig, LRSchedulerTypeUnion, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import rank_zero_warn, WarningCache

_DEEPSPEED_AVAILABLE: bool = _RequirementAvailable("deepspeed")
if _DEEPSPEED_AVAILABLE:
    import deepspeed


class LightningDeepSpeedModule(_LightningModuleWrapperBase):
    def __init__(
        self, pl_module: Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase], precision: Union[str, int]
    ) -> None:
        super().__init__(pl_module)
        self.precision = precision

    def forward(self, *inputs, **kwargs):
        inputs = apply_to_collection(inputs, Tensor, function=self._batch_to)
        return super().forward(*inputs, **kwargs)

    def _batch_to(self, batch: Tensor) -> Tensor:
        if torch.is_floating_point(batch):
            if self.precision == PrecisionType.HALF:
                return batch.half()
            elif self.precision == PrecisionType.BFLOAT:
                return batch.bfloat16()
        return batch


class HPUDeepSpeedStrategy(HPUParallelStrategy):
    strategy_name = "hpu_deepspeed"
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        zero_optimization: bool = True,
        stage: int = 1,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        config: Optional[Union[Path, str, dict]] = None,
        logging_level: int = logging.WARN,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = "hccl",
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models on the Habana® Gaudi® infrastructure.

        `For more information: https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide.html`.
        """
        if not _DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the `HPUDeepSpeedStrategy`, you must have DeepSpeed installed."
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
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._init_deepspeed_distributed()

    def setup(self, trainer: "pl.Trainer") -> None:
        self.accelerator.setup(trainer)
        # we set the device so that optimizers can be created with distributed comms.
        self.lightning_module._device = self.root_device
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        optimizers_to_device(self.optimizers, self.root_device)
        self.init_deepspeed()
        self.barrier()

    def _init_deepspeed_distributed(self) -> None:
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
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to
        """
        if trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.TUNING):
            return
        # Skip initializing optimizers here as DeepSpeed handles optimizers via config.
        # User may have specified config options instead in configure_optimizers, but this is handled
        # via `_initialize_deepspeed_train`
        # empty optimizers, schedulers and frequencies
        self.optimizers = []
        self.lr_scheduler_configs = []
        self.optimizer_frequencies = []

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return True

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
        self.model, optimizer = self._setup_model_and_optimizer(model, optimizers[0])
        return self.model, [optimizer]

    def _setup_model_and_optimizer(
        self, model: Module, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
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

        if not isinstance(self.accelerator, HPUAccelerator):
            raise MisconfigurationException(
                f"HPUDeepSpeed strategy is only supported on GPU but `{self.accelerator.__class__.__name__}` is used."
            )

        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "DeepSpeed currently does not support different `accumulate_grad_batches` at different epochs."
            )

        model = LightningDeepSpeedModule(pl_module=self.model, precision=self.precision_plugin.precision)

        if self.lightning_module.trainer and self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

    def _init_optimizers(self) -> Tuple[Optimizer, Optional[LRSchedulerConfig], Optional[int]]:
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

    def _create_default_config(
        self,
        zero_optimization: bool,
        **zero_kwargs,
    ):
        cfg = {}
        if zero_optimization:
            zero_config = zero_kwargs
            cfg = {
                "zero_optimization": zero_config,
                **cfg,
            }
        return cfg

    @property
    def zero_stage_3(self) -> bool:
        return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3

    def _initialize_deepspeed_train(self, model):
        optimizer, scheduler = None, None
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

    @property
    def deepspeed_engine(self):
        return self.model

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

        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint")

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        # Rely on deepspeed to load the checkpoint and necessary information
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

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the states in `load_checkpoint()`
        pass

    @property
    def lightning_module(self):
        # the model may not be wrapped with DeepEngine & LightningDeepSpeedModule if calling this too early
        module = getattr(self.model, "module", self.model)
        return module.module if isinstance(module, LightningDeepSpeedModule) else module

    @property
    def _multi_device(self) -> bool:
        return self.num_processes > 1 or self.num_nodes > 1

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
