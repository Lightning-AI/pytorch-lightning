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

import logging
import os
from typing import List, Optional, Sequence, Union
from weakref import proxy

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.accelerators.ipu import IPUAccelerator
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from pytorch_lightning.plugins import (
    ApexMixedPrecisionPlugin,
    CheckpointIO,
    DataParallelStrategy,
    DDP2Strategy,
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedPrecisionPlugin,
    DeepSpeedStrategy,
    DoublePrecisionPlugin,
    FullyShardedNativeMixedPrecisionPlugin,
    HorovodStrategy,
    IPUPrecisionPlugin,
    IPUStrategy,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    SingleDeviceStrategy,
    SingleTPUStrategy,
    Strategy,
    TPUBf16PrecisionPlugin,
    TPUPrecisionPlugin,
    TPUSpawnStrategy,
    TrainingTypePluginsRegistry,
)
from pytorch_lightning.plugins.environments import (
    ClusterEnvironment,
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from pytorch_lightning.utilities import (
    _AcceleratorType,
    _StrategyType,
    AMPType,
    device_parser,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import (
    _HOROVOD_AVAILABLE,
    _IPU_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_8,
    _TPU_AVAILABLE,
)

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

log = logging.getLogger(__name__)


class AcceleratorConnector:
    def __init__(
        self,
        num_processes,
        devices,
        tpu_cores,
        ipus,
        accelerator,
        strategy: Optional[Union[str, Strategy]],
        gpus,
        gpu_ids,
        num_nodes,
        sync_batchnorm,
        benchmark,
        replace_sampler_ddp,
        deterministic: bool,
        precision,
        amp_type,
        amp_level,
        plugins,
    ):
        # initialization
        self._device_type = _AcceleratorType.CPU
        self._distrib_type = None
        self._accelerator_type = None

        self.strategy = strategy.lower() if isinstance(strategy, str) else strategy
        # TODO: Rename this to something else once all the distributed flags are moved to strategy
        self.distributed_backend = accelerator

        self._init_deterministic(deterministic)

        self.num_processes = num_processes
        self.devices = devices
        # `gpus` is the input passed to the Trainer, whereas `gpu_ids` is a list of parsed gpu ids.
        self.gpus = gpus
        self.parallel_device_ids = gpu_ids
        self.tpu_cores = tpu_cores
        self.ipus = ipus
        self.num_nodes = num_nodes
        self.sync_batchnorm = sync_batchnorm
        self.benchmark = benchmark
        self.replace_sampler_ddp = replace_sampler_ddp
        if not PrecisionType.supported_type(precision):
            raise MisconfigurationException(
                f"Precision {repr(precision)} is invalid. Allowed precision values: {PrecisionType.supported_types()}"
            )
        self.precision = precision
        self.amp_type = amp_type.lower() if isinstance(amp_type, str) else None
        self.amp_level = amp_level

        self._precision_plugin: Optional[PrecisionPlugin] = None
        self._training_type_plugin: Optional[Strategy] = None
        self._cluster_environment: Optional[ClusterEnvironment] = None
        self._checkpoint_io: Optional[CheckpointIO] = None

        plugins = plugins if plugins is not None else []

        if isinstance(plugins, str):
            plugins = [plugins]

        if not isinstance(plugins, Sequence):
            plugins = [plugins]

        self.plugins = plugins

        self._handle_accelerator_and_strategy()

        self._validate_accelerator_and_devices()

        self._warn_if_devices_flag_ignored()

        self.select_accelerator_type()

        if self.strategy is not None:
            self._set_training_type_plugin()
        else:
            self.set_distributed_mode()

        self.handle_given_plugins()
        self._set_distrib_type_if_training_type_plugin_passed()

        self._cluster_environment = self.select_cluster_environment()

        self.update_device_type_if_ipu_plugin()
        self.update_device_type_if_training_type_plugin_passed()

        self._validate_accelerator_type()
        self._set_devices_if_none()

        self.training_type_plugin = self.final_training_type_plugin()
        self.accelerator = self.training_type_plugin.accelerator
        self._check_plugin_compatibility()

        # benchmarking
        # TODO: should this be moved to GPU accelerator?
        torch.backends.cudnn.benchmark = self.benchmark

        self.replace_sampler_ddp = replace_sampler_ddp

    def _init_deterministic(self, deterministic: bool) -> None:
        self.deterministic = deterministic
        if _TORCH_GREATER_EQUAL_1_8:
            torch.use_deterministic_algorithms(deterministic)
        else:
            torch.set_deterministic(deterministic)
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)
            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def select_accelerator_type(self) -> None:
        if self.distributed_backend == "auto":
            if self.has_tpu:
                self._accelerator_type = _AcceleratorType.TPU
            elif self.has_ipu:
                self._accelerator_type = _AcceleratorType.IPU
            elif self.has_gpu:
                self._accelerator_type = _AcceleratorType.GPU
            else:
                self._set_devices_to_cpu_num_processes()
                self._accelerator_type = _AcceleratorType.CPU
        elif self.distributed_backend == _AcceleratorType.TPU:
            if not self.has_tpu:
                msg = "TPUs are not available" if not _TPU_AVAILABLE else "you didn't pass `tpu_cores` to `Trainer`"
                raise MisconfigurationException(f"You passed `accelerator='tpu'`, but {msg}.")
            self._accelerator_type = _AcceleratorType.TPU
        elif self.distributed_backend == _AcceleratorType.IPU:
            if not self.has_ipu:
                msg = "IPUs are not available" if not _IPU_AVAILABLE else "you didn't pass `ipus` to `Trainer`"
                raise MisconfigurationException(f"You passed `accelerator='ipu'`, but {msg}.")
            self._accelerator_type = _AcceleratorType.IPU
        elif self.distributed_backend == _AcceleratorType.GPU:
            if not self.has_gpu:
                msg = "you didn't pass `gpus` to `Trainer`" if torch.cuda.is_available() else "GPUs are not available"
                raise MisconfigurationException(f"You passed `accelerator='gpu'`, but {msg}.")
            self._accelerator_type = _AcceleratorType.GPU
        elif self.distributed_backend == _AcceleratorType.CPU:
            self._set_devices_to_cpu_num_processes()
            self._accelerator_type = _AcceleratorType.CPU

        if self.distributed_backend in self.accelerator_types:
            self.distributed_backend = None

    def _validate_accelerator_and_devices(self) -> None:
        if self.distributed_backend not in self.accelerator_types and self.devices is not None:
            raise MisconfigurationException(
                f"You passed `devices={self.devices}` but haven't specified"
                " `accelerator=('auto'|'tpu'|'gpu'|'ipu'|'cpu')` for the devices mapping,"
                f" got `accelerator={self.distributed_backend!r}`."
            )

    def _validate_accelerator_type(self) -> None:
        if self._accelerator_type and self._accelerator_type != self._device_type:
            # internal error: should not happen.
            raise ValueError(
                f"Mismatch between the requested accelerator type ({self._accelerator_type})"
                f" and assigned device type ({self._device_type})."
            )
        self._accelerator_type = self._device_type

    def _warn_if_devices_flag_ignored(self) -> None:
        if self.devices is None:
            return
        devices_warning = f"The flag `devices={self.devices}` will be ignored, as you have set"
        if self.distributed_backend in ("auto", _AcceleratorType.TPU):
            if self.tpu_cores is not None:
                rank_zero_warn(f"{devices_warning} `tpu_cores={self.tpu_cores}`")
        elif self.distributed_backend in ("auto", _AcceleratorType.IPU):
            if self.ipus is not None:
                rank_zero_warn(f"{devices_warning} `ipus={self.ipus}`")
        elif self.distributed_backend in ("auto", _AcceleratorType.GPU):
            if self.gpus is not None:
                rank_zero_warn(f"{devices_warning} `gpus={self.gpus}`")
        elif self.distributed_backend in ("auto", _AcceleratorType.CPU):
            if self.num_processes != 1:
                rank_zero_warn(f"{devices_warning} `num_processes={self.num_processes}`")

    def _set_devices_if_none(self) -> None:
        if self.devices is not None:
            return
        if self._accelerator_type == _AcceleratorType.TPU:
            self.devices = self.tpu_cores
        elif self._accelerator_type == _AcceleratorType.IPU:
            self.devices = self.ipus
        elif self._accelerator_type == _AcceleratorType.GPU:
            self.devices = self.gpus
        elif self._accelerator_type == _AcceleratorType.CPU:
            self.devices = self.num_processes

    def _handle_accelerator_and_strategy(self) -> None:
        deprecated_types = [t for t in _StrategyType if t not in (_StrategyType.TPU_SPAWN, _StrategyType.DDP_CPU)]
        if self.distributed_backend is not None and self.distributed_backend in deprecated_types:
            rank_zero_deprecation(
                f"Passing `Trainer(accelerator={self.distributed_backend!r})` has been deprecated"
                f" in v1.5 and will be removed in v1.7. Use `Trainer(strategy={self.distributed_backend!r})` instead."
            )
            if self.strategy is not None:
                raise MisconfigurationException(
                    f"You have passed `Trainer(strategy={self.strategy!r})` but have"
                    f" also passed `Trainer(accelerator={self.distributed_backend!r})`."
                    f" HINT: Use just `Trainer(strategy={self.strategy!r})` instead."
                )
        if self.strategy == _StrategyType.TPU_SPAWN:
            raise MisconfigurationException(
                "`Trainer(strategy='tpu_spawn')` is not a valid strategy,"
                " you can use `Trainer(strategy='ddp_spawn', accelerator='tpu')` instead."
            )
        if self.strategy == _StrategyType.DDP_CPU:
            raise MisconfigurationException(
                "`Trainer(strategy='ddp_cpu')` is not a valid strategy,"
                " you can use `Trainer(strategy='ddp'|'ddp_spawn', accelerator='cpu')` instead."
            )

    def _set_training_type_plugin(self) -> None:
        if isinstance(self.strategy, str) and self.strategy in TrainingTypePluginsRegistry:
            self._training_type_plugin = TrainingTypePluginsRegistry.get(self.strategy)
        if isinstance(self.strategy, str):
            self.set_distributed_mode(self.strategy)
        elif isinstance(self.strategy, Strategy):
            self._training_type_plugin = self.strategy

    def handle_given_plugins(self) -> None:

        for plug in self.plugins:
            if self.strategy is not None and self._is_plugin_training_type(plug):
                raise MisconfigurationException(
                    f"You have passed `Trainer(strategy={self.strategy!r})`"
                    f" and you can only specify one training type plugin, but you have passed {plug} as a plugin."
                )
            if self._is_plugin_training_type(plug):
                rank_zero_deprecation(
                    f"Passing {plug} `strategy` to the `plugins` flag in Trainer has been deprecated"
                    f" in v1.5 and will be removed in v1.7. Use `Trainer(strategy={plug})` instead."
                )

        training_type = self._training_type_plugin or None
        checkpoint = None
        precision = None
        cluster_environment = None

        for plug in self.plugins:
            if isinstance(plug, str) and plug in TrainingTypePluginsRegistry:
                if training_type is None:
                    training_type = TrainingTypePluginsRegistry.get(plug)
                else:
                    raise MisconfigurationException(
                        "You can only specify one precision and one training type plugin."
                        " Found more than 1 training type plugin:"
                        f' {TrainingTypePluginsRegistry[plug]["plugin"]} registered to {plug}'
                    )
            if isinstance(plug, str):
                # Reset the distributed type as the user has overridden training type
                # via the plugins argument
                self._distrib_type = None
                self.set_distributed_mode(plug)

            elif isinstance(plug, Strategy):
                if training_type is None:
                    training_type = plug

                else:
                    raise MisconfigurationException(
                        "You can only specify one training type plugin."
                        f" Available: {type(training_type).__name__}, given: {type(plug).__name__}"
                    )
            elif isinstance(plug, PrecisionPlugin):
                if precision is None:
                    precision = plug
                else:
                    raise MisconfigurationException(
                        "You can only specify one precision plugin."
                        f" Available: {type(precision).__name__}, given: {type(plug).__name__}"
                    )
            elif isinstance(plug, CheckpointIO):
                if checkpoint is None:
                    checkpoint = plug
                else:
                    raise MisconfigurationException(
                        "You can only specify one checkpoint plugin."
                        f" Available: {type(checkpoint).__name__}, given: {type(plug).__name__}"
                    )
            elif isinstance(plug, ClusterEnvironment):
                if cluster_environment is None:
                    cluster_environment = plug
                else:
                    raise MisconfigurationException(
                        "You can only specify one cluster environment. Found more than 1 cluster environment plugin"
                    )
            else:
                raise MisconfigurationException(
                    f"Found invalid type for plugin {plug}. Expected a precision or training type plugin."
                )

        self._training_type_plugin = training_type
        self._precision_plugin = precision
        self._checkpoint_io = checkpoint
        self._cluster_environment = cluster_environment

    @property
    def accelerator_types(self) -> List[str]:
        return ["auto"] + list(_AcceleratorType)

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        if self._precision_plugin is None:
            self._precision_plugin = self.select_precision_plugin()
        return self._precision_plugin

    def final_training_type_plugin(self) -> Strategy:
        if self._training_type_plugin is None:
            self._training_type_plugin = self.select_training_type_plugin()
        self._training_type_plugin = self.resolve_training_type_plugin(self._training_type_plugin)
        # attach checkpoint plugin to the training type plugin
        if self._checkpoint_io is not None:
            self._training_type_plugin.checkpoint_io = self._checkpoint_io
        if (isinstance(self.strategy, Strategy) and self.strategy._precision_plugin is None) or not isinstance(
            self.strategy, Strategy
        ):
            precision_plugin = self.precision_plugin
            if precision_plugin is not None:
                self._training_type_plugin.precision_plugin = precision_plugin
        if (isinstance(self.strategy, Strategy) and self.strategy.accelerator is None) or not isinstance(
            self.strategy, Strategy
        ):
            self._training_type_plugin.accelerator = self.select_accelerator()
        return self._training_type_plugin

    @property
    def cluster_environment(self) -> ClusterEnvironment:
        if self._cluster_environment is None:
            self._cluster_environment = self.select_cluster_environment()
        return self._cluster_environment

    @property
    def has_cpu(self) -> bool:
        return True

    @property
    def use_cpu(self) -> bool:
        return self._accelerator_type == _AcceleratorType.CPU

    @property
    def has_gpu(self) -> bool:
        # Here, we are not checking for GPU availability, but instead if User has passed
        # `gpus` to Trainer for training.
        gpus = self.parallel_device_ids
        if gpus is not None and len(gpus) > 0:
            return True
        return self._map_devices_to_accelerator(_AcceleratorType.GPU)

    @property
    def use_gpu(self) -> bool:
        return self._accelerator_type == _AcceleratorType.GPU and self.has_gpu

    @property
    def has_tpu(self) -> bool:
        # Here, we are not checking for TPU availability, but instead if User has passed
        # `tpu_cores` to Trainer for training.
        if self.tpu_cores is not None:
            return True
        return self._map_devices_to_accelerator(_AcceleratorType.TPU)

    @property
    def use_tpu(self) -> bool:
        return self._accelerator_type == _AcceleratorType.TPU and self.has_tpu

    @property
    def tpu_id(self) -> Optional[int]:
        if self.use_tpu and isinstance(self.tpu_cores, list):
            return self.tpu_cores[0]
        return None

    @property
    def has_ipu(self) -> bool:
        # Here, we are not checking for IPU availability, but instead if User has passed
        # `ipus` to Trainer for training.
        if self.ipus is not None or isinstance(self._training_type_plugin, IPUStrategy):
            return True
        return self._map_devices_to_accelerator(_AcceleratorType.IPU)

    @property
    def use_ipu(self) -> bool:
        return self._accelerator_type == _AcceleratorType.IPU and self.has_ipu

    def _set_devices_to_cpu_num_processes(self) -> None:
        if self.num_processes == 1:
            self._map_devices_to_accelerator(_AcceleratorType.CPU)

    def _map_devices_to_accelerator(self, accelerator: str) -> bool:
        if self.devices is None:
            return False
        if accelerator == _AcceleratorType.TPU and _TPU_AVAILABLE:
            if self.devices == "auto":
                self.devices = TPUAccelerator.auto_device_count()
            self.tpu_cores = device_parser.parse_tpu_cores(self.devices)
            return True
        if accelerator == _AcceleratorType.IPU and _IPU_AVAILABLE:
            if self.devices == "auto":
                self.devices = IPUAccelerator.auto_device_count()
            self.ipus = self.devices
            return True
        if accelerator == _AcceleratorType.GPU and torch.cuda.is_available():
            if self.devices == "auto":
                self.devices = GPUAccelerator.auto_device_count()
            self.gpus = self.devices
            self.parallel_device_ids = device_parser.parse_gpu_ids(self.devices)
            return True
        if accelerator == _AcceleratorType.CPU:
            if self.devices == "auto":
                self.devices = CPUAccelerator.auto_device_count()
            if not isinstance(self.devices, int):
                raise MisconfigurationException(
                    "The flag `devices` must be an int with `accelerator='cpu'`,"
                    f" got `devices={self.devices}` instead."
                )
            self.num_processes = self.devices
            return True
        return False

    @property
    def use_dp(self) -> bool:
        return self._distrib_type == _StrategyType.DP

    @property
    def use_ddp(self) -> bool:
        return self._distrib_type in (
            _StrategyType.DDP,
            _StrategyType.DDP_SPAWN,
            _StrategyType.DDP_SHARDED,
            _StrategyType.DDP_SHARDED_SPAWN,
            _StrategyType.DDP_FULLY_SHARDED,
            _StrategyType.DEEPSPEED,
            _StrategyType.TPU_SPAWN,
        )

    @property
    def use_ddp2(self) -> bool:
        return self._distrib_type == _StrategyType.DDP2

    @property
    def use_horovod(self) -> bool:
        return self._distrib_type == _StrategyType.HOROVOD

    @property
    def use_deepspeed(self) -> bool:
        return self._distrib_type == _StrategyType.DEEPSPEED

    @property
    def _is_sharded_training_type(self) -> bool:
        return isinstance(self._training_type_plugin, (DDPShardedStrategy, DDPSpawnShardedStrategy))

    @property
    def _is_fully_sharded_training_type(self) -> bool:
        return isinstance(self._training_type_plugin, DDPFullyShardedStrategy)

    @property
    def is_distributed(self) -> bool:
        # Used for custom plugins.
        # Custom plugins should implement is_distributed property.
        if hasattr(self.training_type_plugin, "is_distributed") and not self.use_tpu:
            return self.training_type_plugin.is_distributed
        is_distributed = self.use_ddp or self.use_ddp2 or self.use_horovod
        if self.use_tpu:
            is_distributed |= self.training_type_plugin.is_distributed
        return is_distributed

    @property
    def num_gpus(self) -> int:
        gpus = self.parallel_device_ids
        if gpus is None:
            return 0
        return len(gpus)

    @property
    def num_ipus(self) -> int:
        if isinstance(self.ipus, int):
            return self.ipus
        if isinstance(self._training_type_plugin, IPUStrategy):
            return self._training_type_plugin.replication_factor
        return 0

    @property
    def parallel_devices(self) -> List[Union[torch.device, int]]:
        if self.use_gpu:
            devices = [torch.device("cuda", i) for i in self.parallel_device_ids]
        elif self.use_tpu:
            # explicitly don't make a tpu device here!
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/3169
            if isinstance(self.tpu_cores, int):
                devices = list(range(self.tpu_cores))
        elif self.use_ipu:
            devices = list(range(self.num_ipus))
        else:
            devices = [torch.device("cpu")] * self.num_processes
        return devices

    @property
    def root_gpu(self) -> Optional[int]:
        return (
            self.training_type_plugin.root_device.index
            if not isinstance(self.accelerator, (IPUAccelerator, TPUAccelerator))
            else None
        )

    @staticmethod
    def _is_plugin_training_type(plugin: Union[str, Strategy]) -> bool:
        if isinstance(plugin, str) and (plugin in TrainingTypePluginsRegistry or plugin in list(_StrategyType)):
            return True
        return isinstance(plugin, Strategy)

    @property
    def is_training_type_in_plugins(self) -> bool:
        return any(
            (isinstance(plug, str) and plug in TrainingTypePluginsRegistry) or isinstance(plug, Strategy)
            for plug in self.plugins
        )

    def select_precision_plugin(self) -> PrecisionPlugin:
        # set precision type
        self.amp_type = AMPType.from_str(self.amp_type)

        # validation for all plugins
        if self.amp_level is not None and self.amp_type != AMPType.APEX:
            raise MisconfigurationException(
                f"You have asked for `amp_level={self.amp_level!r}` but it's only supported with `amp_backend='apex'`."
            )

        if self.use_ipu:
            if self.precision not in (16, 32):
                raise MisconfigurationException(
                    f"`Trainer(accelerator='ipu', precision={self.precision!r})` is not supported."
                )
            return IPUPrecisionPlugin(self.precision)
        if self.use_tpu:
            if self.precision == 32:
                return TPUPrecisionPlugin()
            elif self.precision == 64:
                raise MisconfigurationException(
                    "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                    " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`"
                    " requesting this feature."
                )
            elif self.precision in (16, "bf16"):
                if self.precision == 16:
                    # this is not deprecated to ease transition between accelerator environments
                    rank_zero_warn(
                        f"You passed `Trainer(accelerator='tpu', precision=16)` but {self.amp_type.value} AMP"
                        f" is not supported with TPUs. Using `precision='bf16'` instead."
                    )
                return TPUBf16PrecisionPlugin()

        if self._distrib_type == _StrategyType.DEEPSPEED or isinstance(self._training_type_plugin, DeepSpeedStrategy):
            return DeepSpeedPrecisionPlugin(self.precision, self.amp_type, self.amp_level)

        if self.precision == 32:
            return PrecisionPlugin()
        if self.precision == 64:
            return DoublePrecisionPlugin()

        # maybe convert the precision value
        if self.precision == 16 and self.use_cpu:
            if self.amp_type == AMPType.APEX:
                # apex was explicitly passed, not a good idea to silently switch to native AMP
                raise MisconfigurationException(
                    "You passed `Trainer(accelerator='cpu', precision=16, amp_type='apex')`"
                    " but apex AMP not supported on CPU."
                )
            # this automatic switch is to ease transition between accelerator environments
            rank_zero_warn(
                "You passed `Trainer(accelerator='cpu', precision=16)` but native AMP is not supported on CPU."
                " Using `precision='bf16'` instead."
            )
            self.precision = "bf16"

        if self.precision in (16, "bf16"):
            if self.precision == "bf16" and self.amp_type != AMPType.NATIVE:
                raise MisconfigurationException(
                    f"You passed `Trainer(amp_type={self.amp_type.value!r}, precision='bf16')` but it's not supported."
                    " Try using `amp_type='native'` instead."
                )

            rank_zero_info(
                f"Using 16bit {self.amp_type.value} Automatic Mixed Precision (AMP)"
                if self.precision == 16
                else "Using bfloat16 Automatic Mixed Precision (AMP)"
            )

            if self.amp_type == AMPType.NATIVE:
                device = "cpu" if self.use_cpu else "cuda"

                if self._is_sharded_training_type:
                    return ShardedNativeMixedPrecisionPlugin(self.precision, device)
                if self._is_fully_sharded_training_type:
                    return FullyShardedNativeMixedPrecisionPlugin(self.precision, device)
                return NativeMixedPrecisionPlugin(self.precision, device)

            if self.amp_type == AMPType.APEX:
                if self._is_sharded_training_type or self._is_fully_sharded_training_type:
                    raise MisconfigurationException(
                        "Sharded plugins are not supported with apex, please switch to `amp_backend='native'`."
                    )
                self.amp_level = self.amp_level or "O2"
                return ApexMixedPrecisionPlugin(self.amp_level)

        raise RuntimeError("No precision set")

    def select_training_type_plugin(self) -> Strategy:
        if (
            isinstance(self.distributed_backend, Accelerator)
            and self.distributed_backend.training_type_plugin is not None
        ):
            plugin = self.distributed_backend.training_type_plugin
        elif self.use_ddp2:
            plugin = DDP2Strategy(parallel_devices=self.parallel_devices, cluster_environment=self.cluster_environment)
        elif self.use_ddp and self.use_deepspeed:
            plugin = DeepSpeedStrategy(
                cluster_environment=self.select_cluster_environment(), parallel_devices=self.parallel_devices
            )
        elif self.use_ddp:
            use_slurm_ddp = self.use_ddp and self._is_slurm_managing_tasks()
            use_torchelastic_ddp = self.use_ddp and TorchElasticEnvironment.detect()
            use_kubeflow_ddp = self.use_ddp and KubeflowEnvironment.detect()
            use_ddp_spawn = self._distrib_type == _StrategyType.DDP_SPAWN
            use_ddp_cpu_spawn = use_ddp_spawn and self.use_cpu
            use_tpu_spawn = self.use_tpu and self._distrib_type == _StrategyType.TPU_SPAWN
            use_ddp_cpu_torch_elastic = use_ddp_cpu_spawn and TorchElasticEnvironment.detect()
            use_ddp_cpu_kubeflow = use_ddp_cpu_spawn and KubeflowEnvironment.detect()
            use_ddp_cpu_slurm = use_ddp_cpu_spawn and self._is_slurm_managing_tasks()
            use_ddp_sharded = self._distrib_type == _StrategyType.DDP_SHARDED
            use_ddp_sharded_spawn = self._distrib_type == _StrategyType.DDP_SHARDED_SPAWN
            use_ddp_fully_sharded = self._distrib_type == _StrategyType.DDP_FULLY_SHARDED

            if use_tpu_spawn:
                ddp_strategy_cls = TPUSpawnStrategy
            elif use_ddp_sharded:
                ddp_strategy_cls = DDPShardedStrategy
            elif use_ddp_sharded_spawn:
                ddp_strategy_cls = DDPSpawnShardedStrategy
            elif (
                use_ddp_cpu_slurm
                or use_slurm_ddp
                or use_ddp_cpu_torch_elastic
                or use_torchelastic_ddp
                or use_kubeflow_ddp
                or use_ddp_cpu_kubeflow
            ):
                ddp_strategy_cls = DDPStrategy
            elif use_ddp_spawn or use_ddp_cpu_spawn:
                ddp_strategy_cls = DDPSpawnStrategy
            elif use_ddp_fully_sharded:
                ddp_strategy_cls = DDPFullyShardedStrategy
            else:
                ddp_strategy_cls = DDPStrategy

            plugin = ddp_strategy_cls(
                parallel_devices=self.parallel_devices, cluster_environment=self.cluster_environment
            )
        elif self.use_dp:
            plugin = DataParallelStrategy(parallel_devices=self.parallel_devices)
        elif self.use_horovod:
            plugin = HorovodStrategy(parallel_devices=self.parallel_devices)
        elif self.use_tpu and isinstance(self.tpu_cores, list):
            plugin = SingleTPUStrategy(self.tpu_id)
        elif self.use_ipu:
            plugin = IPUStrategy(parallel_devices=self.parallel_devices)
        else:
            single_gpu_ordinal = device_parser.determine_root_gpu_device(self.parallel_device_ids)
            plugin = SingleDeviceStrategy(device=torch.device(f"cuda:{single_gpu_ordinal}" if self.use_gpu else "cpu"))
        return plugin

    def resolve_training_type_plugin(self, training_type: Strategy) -> Strategy:
        # necessary for when the user has passed in a plugin
        if hasattr(training_type, "parallel_devices") and getattr(training_type, "parallel_devices") is None:
            training_type.parallel_devices = self.parallel_devices
            if hasattr(training_type, "num_processes"):
                training_type.num_processes = len(self.parallel_devices)

        if hasattr(training_type, "cluster_environment") and getattr(training_type, "cluster_environment") is None:
            # transfer ownership of the cluster environment to the training type
            training_type.cluster_environment = self.cluster_environment
            self._cluster_environment = proxy(self.cluster_environment)

        if hasattr(training_type, "num_nodes"):
            # set num_nodes for training_type from trainer setting
            training_type.num_nodes = self.num_nodes

        if hasattr(training_type, "sync_batchnorm"):
            # set sync_batchnorm for training_type from trainer setting
            training_type.sync_batchnorm = self.sync_batchnorm

        return training_type

    def select_accelerator(self) -> Accelerator:
        if isinstance(self.distributed_backend, Accelerator):
            # custom accelerator from user
            if self._precision_plugin is not None or self._training_type_plugin is not None:
                # plugins also specified by user
                rank_zero_warn(
                    "Specified `Precision` and `TrainingType` plugins will be ignored,"
                    " since an `Accelerator` instance was provided."
                )
            return self.distributed_backend

        if self.use_gpu:
            acc_cls = GPUAccelerator
        elif self.use_tpu:
            acc_cls = TPUAccelerator
        elif self.use_ipu:
            acc_cls = IPUAccelerator
        else:
            acc_cls = CPUAccelerator

        accelerator = acc_cls()
        return accelerator

    def select_cluster_environment(self) -> ClusterEnvironment:
        if self._cluster_environment is not None:
            return self._cluster_environment
        if self._is_slurm_managing_tasks():
            rank_zero_info("Multiprocessing is handled by SLURM.")
            return SLURMEnvironment()

        for env_type in (TorchElasticEnvironment, KubeflowEnvironment, LSFEnvironment):
            if env_type.detect():
                return env_type()

        return LightningEnvironment()

    def set_distributed_mode(self, strategy: Optional[str] = None):

        if strategy is None and self.is_training_type_in_plugins:
            return

        if strategy is not None and strategy in TrainingTypePluginsRegistry:
            self.distributed_backend = TrainingTypePluginsRegistry[strategy]["distributed_backend"]
        elif strategy is not None:
            self.distributed_backend = strategy

        if isinstance(self.distributed_backend, Accelerator):
            return

        is_cpu_accelerator_type = self._accelerator_type and self._accelerator_type == _AcceleratorType.CPU
        _use_cpu = is_cpu_accelerator_type or self.distributed_backend and "cpu" in self.distributed_backend

        if self.distributed_backend is None:
            if self.has_horovodrun():
                self._set_horovod_backend()
            elif self.num_gpus == 0 and self.num_nodes > 1:
                self._distrib_type = _StrategyType.DDP
            elif self.num_gpus == 0 and self.num_processes > 1:
                self.distributed_backend = _StrategyType.DDP_SPAWN
            elif self.num_gpus > 1 and not _use_cpu:
                rank_zero_warn(
                    "You requested multiple GPUs but did not specify a backend, e.g."
                    ' `Trainer(strategy="dp"|"ddp"|"ddp2")`. Setting `strategy="ddp_spawn"` for you.'
                )
                self.distributed_backend = _StrategyType.DDP_SPAWN

        # special case with DDP on CPUs
        if self.distributed_backend == _StrategyType.DDP_CPU:
            if _TPU_AVAILABLE:
                raise MisconfigurationException(
                    "`accelerator='ddp_cpu'` is not supported on TPU machines. "
                    "Learn more: https://github.com/PyTorchLightning/pytorch-lightning/issues/7810"
                )
            if self.num_processes == 1 and self.num_nodes > 1:
                self._distrib_type = _StrategyType.DDP
            else:
                self._distrib_type = _StrategyType.DDP_SPAWN
            if self.num_gpus > 0:
                rank_zero_warn(
                    "You requested one or more GPUs, but set `accelerator='ddp_cpu'`. Training will not use GPUs."
                )
                self.parallel_device_ids = None
            if self.num_processes is None:
                # define the max CPU available
                self.num_processes = os.cpu_count()
        # special case with TPUs
        elif self.has_tpu and not _use_cpu:
            self._device_type = _AcceleratorType.TPU
            if isinstance(self.tpu_cores, int):
                self._distrib_type = _StrategyType.TPU_SPAWN
        elif self.has_ipu and not _use_cpu:
            self._device_type = _AcceleratorType.IPU
        elif self.distributed_backend and self._distrib_type is None:
            self._distrib_type = _StrategyType(self.distributed_backend)

        if self.num_gpus > 0 and not _use_cpu:
            self._device_type = _AcceleratorType.GPU

        _gpu_distrib_types = (_StrategyType.DP, _StrategyType.DDP, _StrategyType.DDP_SPAWN, _StrategyType.DDP2)
        # DP and DDP2 cannot run without GPU
        if self.num_gpus == 0 and self._distrib_type in _gpu_distrib_types and not _use_cpu:

            if (self.num_nodes and self.num_nodes > 1) or (self.num_processes and self.num_processes > 1):
                if self._distrib_type in (_StrategyType.DP, _StrategyType.DDP2):
                    rank_zero_warn(
                        f"{self._distrib_type.value!r} is not supported on CPUs, hence setting `strategy='ddp'`."
                    )
                    self._distrib_type = _StrategyType.DDP
            else:
                rank_zero_warn("You are running on single node with no parallelization, so distributed has no effect.")
                self._distrib_type = None

        # finished configuring self._distrib_type, check ipython environment
        self.check_interactive_compatibility()

        # for DDP overwrite nb processes by requested GPUs
        if self._device_type == _AcceleratorType.GPU and self._distrib_type in (
            _StrategyType.DDP,
            _StrategyType.DDP_SPAWN,
        ):
            self.num_processes = self.num_gpus

        if self._device_type == _AcceleratorType.GPU and self._distrib_type == _StrategyType.DDP2:
            self.num_processes = self.num_nodes

        # Horovod is an extra case...
        if self.distributed_backend == _StrategyType.HOROVOD:
            self._set_horovod_backend()

        using_valid_distributed = self.use_ddp or self.use_ddp2
        if self.num_nodes > 1 and not using_valid_distributed:
            # throw error to force user to choose a supported strategy type such as ddp or ddp2
            raise MisconfigurationException(
                "Your chosen strategy does not support `num_nodes > 1`. Please set `strategy=('ddp'|'ddp2')`."
            )

    def _set_horovod_backend(self):
        self.check_horovod()
        self._distrib_type = _StrategyType.HOROVOD

        # Initialize Horovod to get rank / size info
        hvd.init()
        if self.has_gpu:
            # Horovod assigns one local GPU per process
            self.parallel_device_ids = list(range(hvd.local_size()))
        else:
            self.num_processes = hvd.local_size()

    def check_interactive_compatibility(self):
        """Raises a `MisconfigurationException` if the accelerator and/or plugin is not compatible with an
        interactive environment."""
        from pytorch_lightning.utilities import _IS_INTERACTIVE

        if _IS_INTERACTIVE and self._distrib_type is not None and not self._distrib_type.is_interactive_compatible():
            raise MisconfigurationException(
                f"`Trainer(strategy={self._distrib_type.value!r})` or"
                f" `Trainer(accelerator={self._distrib_type.value!r})` is not compatible with an interactive"
                " environment. Run your code as a script, or choose one of the compatible backends:"
                f" {', '.join(_StrategyType.interactive_compatible_types())}."
                " In case you are spawning processes yourself, make sure to include the Trainer"
                " creation inside the worker function."
            )

    def check_horovod(self):
        """Raises a `MisconfigurationException` if the Trainer is not configured correctly for Horovod."""
        if not _HOROVOD_AVAILABLE:
            raise MisconfigurationException(
                'Requested `accelerator="horovod"`, but Horovod is not installed.'
                "Install with \n $HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]"
            )

        if self.num_gpus > 1 or self.num_nodes > 1:
            raise MisconfigurationException(
                "Horovod does not support setting num_nodes / num_gpus explicitly. Use "
                "horovodrun / mpirun to configure the number of processes."
            )

    @staticmethod
    def has_horovodrun() -> bool:
        """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
        return _HOROVOD_AVAILABLE and ("OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ)

    def update_device_type_if_ipu_plugin(self) -> None:
        # This allows the poptorch.Options that are passed into the IPUStrategy to be the source of truth,
        # which gives users the flexibility to not have to pass `ipus` flag directly to Trainer
        if isinstance(self._training_type_plugin, IPUStrategy) and self._device_type != _AcceleratorType.IPU:
            self._device_type = _AcceleratorType.IPU

    def update_device_type_if_training_type_plugin_passed(self) -> None:
        if isinstance(self.strategy, Strategy) or any(isinstance(plug, Strategy) for plug in self.plugins):
            if self._accelerator_type is not None:
                if self.use_ipu:
                    self._device_type = _AcceleratorType.IPU
                elif self.use_tpu:
                    self._device_type = _AcceleratorType.TPU
                elif self.use_gpu:
                    self._device_type = _AcceleratorType.GPU
            else:
                if self.has_ipu:
                    self._device_type = _AcceleratorType.IPU
                elif self.has_tpu:
                    self._device_type = _AcceleratorType.TPU
                elif self.has_gpu:
                    self._device_type = _AcceleratorType.GPU

    def _set_distrib_type_if_training_type_plugin_passed(self):
        # This is required as when `Strategy` instance is passed to either `strategy`
        # or `plugins` flag, `AcceleratorConnector.set_distributed_mode` is not required to be
        # called and `_distrib_type` is not set.
        if self._distrib_type is not None:
            return
        if self._training_type_plugin is not None:
            self._distrib_type = getattr(self._training_type_plugin, "distributed_backend", None)

    def _is_slurm_managing_tasks(self) -> bool:
        """Returns whether we let SLURM manage the processes or not.

        Returns ``True`` if and only if these conditions match:

            - A SLURM cluster is detected
            - A distributed plugin is being used
            - The process is not launching in interactive mode
            - The number of tasks in SLURM matches the requested number of devices and nodes in the Trainer
        """
        if (
            (not self.use_ddp and not self.use_ddp2)
            or not SLURMEnvironment.detect()
            or SLURMEnvironment.job_name() == "bash"  # in interactive mode we don't manage tasks
        ):
            return False

        total_requested_devices = (self.num_gpus or self.num_processes) * self.num_nodes
        num_slurm_tasks = int(os.environ["SLURM_NTASKS"], 0)
        return num_slurm_tasks == total_requested_devices

    def _check_plugin_compatibility(self) -> None:
        """Checks that selected plugins are compatible with each other.

        Raises:
            ValueError: If an invalid combination of Accelerator, Strategy, PrecisionPlugin is found.
        """
        if isinstance(self.accelerator, TPUAccelerator):
            if not isinstance(self.training_type_plugin.precision_plugin, TPUPrecisionPlugin):
                raise ValueError(
                    f"The `TPUAccelerator` can only be used with a `TPUPrecisionPlugin`,"
                    f" found: {self.training_type_plugin.precision_plugin}."
                )
            if not isinstance(self.training_type_plugin, (SingleTPUStrategy, TPUSpawnStrategy)):
                raise ValueError(
                    "The `TPUAccelerator` can only be used with a `SingleTPUStrategy` or `TPUSpawnStrategy`,"
                    f" found {self.training_type_plugin}."
                )
