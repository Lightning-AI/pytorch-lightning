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
    DataParallelPlugin,
    DDP2Plugin,
    DDPFullyShardedPlugin,
    DDPPlugin,
    DDPShardedPlugin,
    DDPSpawnPlugin,
    DDPSpawnShardedPlugin,
    DeepSpeedPlugin,
    DeepSpeedPrecisionPlugin,
    DoublePrecisionPlugin,
    FullyShardedNativeMixedPrecisionPlugin,
    HorovodPlugin,
    IPUPlugin,
    IPUPrecisionPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    SingleDevicePlugin,
    SingleTPUPlugin,
    TPUHalfPrecisionPlugin,
    TPUSpawnPlugin,
    TrainingTypePlugin,
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
    _APEX_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _IPU_AVAILABLE,
    _NATIVE_AMP_AVAILABLE,
    _TPU_AVAILABLE,
    AMPType,
    device_parser,
    DeviceType,
    DistributedType,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException

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
        distributed_backend,
        accelerator,
        gpus,
        gpu_ids,
        num_nodes,
        sync_batchnorm,
        benchmark,
        replace_sampler_ddp,
        deterministic,
        precision,
        amp_type,
        amp_level,
        plugins,
    ):
        # initialization
        self._device_type = DeviceType.CPU
        self._distrib_type = None
        self._accelerator_type = None

        if distributed_backend is not None:
            rank_zero_deprecation(
                f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
                f" Use `Trainer(accelerator={distributed_backend})` instead."
            )
        distributed_backend = distributed_backend or accelerator

        self.num_processes = num_processes
        self.devices = devices
        # `gpus` is the input passed to the Trainer, whereas `gpu_ids` is a list of parsed gpu ids.
        self.gpus = gpus
        self.parallel_device_ids = gpu_ids
        self.tpu_cores = tpu_cores
        self.ipus = ipus
        self.distributed_backend = distributed_backend
        self.num_nodes = num_nodes
        self.sync_batchnorm = sync_batchnorm
        self.benchmark = benchmark
        self.replace_sampler_ddp = replace_sampler_ddp
        self.deterministic = deterministic
        self.precision = precision
        self.amp_type = amp_type.lower() if isinstance(amp_type, str) else None
        self.amp_level = amp_level
        self.is_slurm_managing_tasks = False

        self._precision_plugin: Optional[PrecisionPlugin] = None
        self._training_type_plugin: Optional[TrainingTypePlugin] = None
        self._cluster_environment: Optional[ClusterEnvironment] = None

        plugins = plugins if plugins is not None else []

        if isinstance(plugins, str):
            plugins = [plugins]

        if not isinstance(plugins, Sequence):
            plugins = [plugins]

        self.plugins = plugins

        self._validate_accelerator_and_devices()
        self._warn_if_devices_flag_ignored()

        self.select_accelerator_type()
        self.set_distributed_mode()
        self.configure_slurm_ddp()

        self.handle_given_plugins()
        self.update_device_type_if_ipu_plugin()

        self._validate_accelerator_type()
        self._set_devices_if_none()

        self._training_type_plugin_resolved = False
        self.accelerator = self.select_accelerator()

        # override dist backend when using tpus
        if self.use_tpu:
            self.distributed_backend = "tpu"

        # init flags for SLURM+DDP to work
        self.world_size = 1
        self.interactive_ddp_procs = []
        self.global_rank = 0

        # benchmarking
        # TODO: should this be moved to GPU accelerator?
        torch.backends.cudnn.benchmark = self.benchmark

        # determinism for cudnn
        # TODO: should this be moved to GPU accelerator?
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

        self.replace_sampler_ddp = replace_sampler_ddp

    def select_accelerator_type(self) -> None:
        if self.distributed_backend == "auto":
            if self.has_tpu:
                self._accelerator_type = DeviceType.TPU
            elif self.has_ipu:
                self._accelerator_type = DeviceType.IPU
            elif self.has_gpu:
                self._accelerator_type = DeviceType.GPU
            else:
                self._set_devices_to_cpu_num_processes()
                self._accelerator_type = DeviceType.CPU
        elif self.distributed_backend == DeviceType.TPU:
            if not self.has_tpu:
                msg = "TPUs are not available" if not _TPU_AVAILABLE else "you didn't pass `tpu_cores` to `Trainer`"
                raise MisconfigurationException(f"You passed `accelerator='tpu'`, but {msg}.")
            self._accelerator_type = DeviceType.TPU
        elif self.distributed_backend == DeviceType.IPU:
            if not self.has_ipu:
                msg = "IPUs are not available" if not _IPU_AVAILABLE else "you didn't pass `ipus` to `Trainer`"
                raise MisconfigurationException(f"You passed `accelerator='ipu'`, but {msg}.")
            self._accelerator_type = DeviceType.IPU
        elif self.distributed_backend == DeviceType.GPU:
            if not self.has_gpu:
                msg = "you didn't pass `gpus` to `Trainer`" if torch.cuda.is_available() else "GPUs are not available"
                raise MisconfigurationException(f"You passed `accelerator='gpu'`, but {msg}.")
            self._accelerator_type = DeviceType.GPU
        elif self.distributed_backend == DeviceType.CPU:
            self._set_devices_to_cpu_num_processes()
            self._accelerator_type = DeviceType.CPU

        if self.distributed_backend in ["auto"] + list(DeviceType):
            self.distributed_backend = None

    def _validate_accelerator_and_devices(self) -> None:
        if self.distributed_backend not in ["auto"] + list(DeviceType) and self.devices is not None:
            raise MisconfigurationException(
                f"You passed `devices={self.devices}` but haven't specified"
                " `accelerator=('auto'|'tpu'|'gpu'|'ipu'|'cpu')` for the devices mapping,"
                f" got `accelerator={self.distributed_backend}`."
            )

    def _validate_accelerator_type(self) -> None:
        if self._accelerator_type and self._accelerator_type != self._device_type:
            raise MisconfigurationException(
                f"Mismatch between the requested accelerator type ({self._accelerator_type})"
                f" and assigned device type ({self._device_type})."
            )
        self._accelerator_type = self._device_type

    def _warn_if_devices_flag_ignored(self) -> None:
        if self.devices is None:
            return
        devices_warning = f"The flag `devices={self.devices}` will be ignored, as you have set"
        if self.distributed_backend == "auto":
            if self.tpu_cores is not None:
                rank_zero_warn(f"{devices_warning} `tpu_cores={self.tpu_cores}`")
            elif self.ipus is not None:
                rank_zero_warn(f"{devices_warning} `ipus={self.ipus}`")
            elif self.gpus is not None:
                rank_zero_warn(f"{devices_warning} `gpus={self.gpus}`")
            elif self.num_processes != 1:
                rank_zero_warn(f"{devices_warning} `num_processes={self.num_processes}`")
        elif self.distributed_backend == DeviceType.TPU:
            if self.tpu_cores is not None:
                rank_zero_warn(f"{devices_warning} `tpu_cores={self.tpu_cores}`")
        elif self.distributed_backend == DeviceType.IPU:
            if self.ipus is not None:
                rank_zero_warn(f"{devices_warning} `ipus={self.ipus}`")
        elif self.distributed_backend == DeviceType.GPU:
            if self.gpus is not None:
                rank_zero_warn(f"{devices_warning} `gpus={self.gpus}`")
        elif self.distributed_backend == DeviceType.CPU:
            if self.num_processes != 1:
                rank_zero_warn(f"{devices_warning} `num_processes={self.num_processes}`")

    def _set_devices_if_none(self) -> None:
        if self.devices is not None:
            return
        if self._accelerator_type == DeviceType.TPU:
            self.devices = self.tpu_cores
        elif self._accelerator_type == DeviceType.IPU:
            self.devices = self.ipus
        elif self._accelerator_type == DeviceType.GPU:
            self.devices = self.gpus
        elif self._accelerator_type == DeviceType.CPU:
            self.devices = self.num_processes

    def handle_given_plugins(self) -> None:

        training_type = None
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

            elif isinstance(plug, TrainingTypePlugin):
                if training_type is None:
                    training_type = plug

                else:
                    raise MisconfigurationException(
                        "You can only specify one precision and one training type plugin."
                        f" Found more than 1 training type plugin: {type(plug).__name__}"
                    )
            elif isinstance(plug, PrecisionPlugin):
                if precision is None:
                    precision = plug
                else:
                    raise MisconfigurationException(
                        "You can only specify one precision and one training type plugin."
                        f" Found more than 1 precision plugin: {type(plug).__name__}"
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
        self._cluster_environment = cluster_environment or self.select_cluster_environment()

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        if self._precision_plugin is None:
            self._precision_plugin = self.select_precision_plugin()
        return self._precision_plugin

    @property
    def training_type_plugin(self) -> TrainingTypePlugin:
        if self._training_type_plugin_resolved:
            # avoid calling `resolve_training_type_plugin` multiple times
            return self._training_type_plugin
        if self._training_type_plugin is None:
            self._training_type_plugin = self.select_training_type_plugin()
        self._training_type_plugin = self.resolve_training_type_plugin(self._training_type_plugin)
        self._training_type_plugin_resolved = True

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
        return self._accelerator_type == DeviceType.CPU

    @property
    def has_gpu(self) -> bool:
        # Here, we are not checking for GPU availability, but instead if User has passed
        # `gpus` to Trainer for training.
        gpus = self.parallel_device_ids
        if gpus is not None and len(gpus) > 0:
            return True
        return self._map_devices_to_accelerator(DeviceType.GPU)

    @property
    def use_gpu(self) -> bool:
        return self._accelerator_type == DeviceType.GPU and self.has_gpu

    @property
    def has_tpu(self) -> bool:
        # Here, we are not checking for TPU availability, but instead if User has passed
        # `tpu_cores` to Trainer for training.
        if self.tpu_cores is not None:
            return True
        return self._map_devices_to_accelerator(DeviceType.TPU)

    @property
    def use_tpu(self) -> bool:
        return self._accelerator_type == DeviceType.TPU and self.has_tpu

    @property
    def tpu_id(self) -> Optional[int]:
        if self.use_tpu and isinstance(self.tpu_cores, list):
            return self.tpu_cores[0]
        return None

    @property
    def has_ipu(self) -> bool:
        # Here, we are not checking for IPU availability, but instead if User has passed
        # `ipus` to Trainer for training.
        if self.ipus is not None or isinstance(self._training_type_plugin, IPUPlugin):
            return True
        return self._map_devices_to_accelerator(DeviceType.IPU)

    @property
    def use_ipu(self) -> bool:
        return self._accelerator_type == DeviceType.IPU and self.has_ipu

    def _set_devices_to_cpu_num_processes(self) -> None:
        if self.num_processes == 1:
            self._map_devices_to_accelerator(DeviceType.CPU)

    def _map_devices_to_accelerator(self, accelerator: str) -> bool:
        if self.devices is None:
            return False
        if accelerator == DeviceType.TPU and _TPU_AVAILABLE:
            self.tpu_cores = device_parser.parse_tpu_cores(self.devices)
            return True
        if accelerator == DeviceType.IPU and _IPU_AVAILABLE:
            self.ipus = self.devices
            return True
        if accelerator == DeviceType.GPU and torch.cuda.is_available():
            self.gpus = self.devices
            self.parallel_device_ids = device_parser.parse_gpu_ids(self.devices)
            return True
        if accelerator == DeviceType.CPU:
            if not isinstance(self.devices, int):
                raise MisconfigurationException(
                    "The flag `devices` only supports integer for `accelerator='cpu'`,"
                    f" got `devices={self.devices}` instead."
                )
            self.num_processes = self.devices
            return True
        return False

    @property
    def use_dp(self) -> bool:
        return self._distrib_type == DistributedType.DP

    @property
    def use_ddp(self) -> bool:
        return self._distrib_type in (
            DistributedType.DDP,
            DistributedType.DDP_SPAWN,
            DistributedType.DDP_SHARDED,
            DistributedType.DDP_SHARDED_SPAWN,
            DistributedType.DDP_FULLY_SHARDED,
            DistributedType.DEEPSPEED,
            DistributedType.TPU_SPAWN,
        )

    @property
    def use_ddp2(self) -> bool:
        return self._distrib_type == DistributedType.DDP2

    @property
    def use_horovod(self) -> bool:
        return self._distrib_type == DistributedType.HOROVOD

    @property
    def use_deepspeed(self) -> bool:
        return self._distrib_type == DistributedType.DEEPSPEED

    @property
    def _is_sharded_training_type(self) -> bool:
        return isinstance(self.training_type_plugin, (DDPShardedPlugin, DDPSpawnShardedPlugin))

    @property
    def _is_fully_sharded_training_type(self) -> bool:
        return isinstance(self.training_type_plugin, DDPFullyShardedPlugin)

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
        if isinstance(self._training_type_plugin, IPUPlugin):
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
            self.accelerator.root_device.index
            if not isinstance(self.accelerator, (IPUAccelerator, TPUAccelerator))
            else None
        )

    @property
    def is_training_type_in_plugins(self) -> bool:
        return any(isinstance(plug, str) and plug in TrainingTypePluginsRegistry for plug in self.plugins)

    @property
    def is_using_torchelastic(self) -> bool:
        """
        .. deprecated:: v1.3
            Will be removed in v1.5.0.
        Returns:
            ``True`` if the current process was launched using the torchelastic command.
        """
        rank_zero_deprecation(
            "The property `AcceleratorConnector.is_using_torchelastic` was deprecated in v1.3"
            " and will be removed in 1.5. Use `TorchElasticEnvironment.is_using_torchelastic()` instead."
        )
        return TorchElasticEnvironment.is_using_torchelastic()

    def select_precision_plugin(self) -> PrecisionPlugin:
        # set precision type
        self.amp_type = AMPType.from_str(self.amp_type)

        if self.use_ipu:
            return IPUPrecisionPlugin(self.precision)

        if self._distrib_type == DistributedType.DEEPSPEED or isinstance(self._training_type_plugin, DeepSpeedPlugin):
            return DeepSpeedPrecisionPlugin(self.precision)

        if self.precision == 32:
            return PrecisionPlugin()
        if self.precision == 64:
            return DoublePrecisionPlugin()
        if self.precision == 16:
            if self.use_tpu:
                return TPUHalfPrecisionPlugin()

            if self.amp_type == AMPType.NATIVE:
                if self.use_cpu:
                    raise MisconfigurationException(
                        "You have asked for native AMP on CPU, but AMP is only available on GPU."
                    )
                if not _NATIVE_AMP_AVAILABLE:
                    msg = (
                        "You have asked for native AMP but your PyTorch version does not support it."
                        " Consider upgrading with `pip install torch>=1.6`."
                    )
                    if _APEX_AVAILABLE:
                        self.amp_type = AMPType.APEX
                        msg += " We will attempt to use NVIDIA Apex for this session."
                        rank_zero_warn(msg)
                    else:
                        raise MisconfigurationException(msg)
                else:
                    log.info("Using native 16bit precision.")
                    if self._is_sharded_training_type:
                        return ShardedNativeMixedPrecisionPlugin()
                    if self._is_fully_sharded_training_type:
                        return FullyShardedNativeMixedPrecisionPlugin()
                    return NativeMixedPrecisionPlugin()

            if self.amp_type == AMPType.APEX:
                if not _APEX_AVAILABLE:
                    raise MisconfigurationException(
                        "You have asked for Apex AMP but you have not installed it yet."
                        " Install apex first using this guide: https://github.com/NVIDIA/apex#linux"
                    )
                if self._is_sharded_training_type or self._is_fully_sharded_training_type:
                    raise MisconfigurationException(
                        "Sharded Plugin is not supported with Apex AMP,"
                        " please using native AMP for 16-bit precision."
                    )
                log.info("Using APEX 16bit precision.")
                return ApexMixedPrecisionPlugin(self.amp_level)

        raise NotImplementedError("We only support precisions 64, 32 and 16!")

    def select_training_type_plugin(self) -> TrainingTypePlugin:
        if (
            isinstance(self.distributed_backend, Accelerator)
            and self.distributed_backend.training_type_plugin is not None
        ):
            plugin = self.distributed_backend.training_type_plugin
        elif self.use_ddp2:
            plugin = DDP2Plugin(parallel_devices=self.parallel_devices, cluster_environment=self.cluster_environment)
        elif self.use_ddp and self.use_deepspeed:
            plugin = DeepSpeedPlugin(
                cluster_environment=self.select_cluster_environment(), parallel_devices=self.parallel_devices
            )
        elif self.use_ddp:
            use_slurm_ddp = self.use_ddp and self.is_slurm_managing_tasks
            use_torchelastic_ddp = self.use_ddp and TorchElasticEnvironment.is_using_torchelastic()
            use_kubeflow_ddp = self.use_ddp and KubeflowEnvironment.is_using_kubeflow()
            use_ddp_spawn = self._distrib_type == DistributedType.DDP_SPAWN
            use_ddp_cpu_spawn = use_ddp_spawn and self.use_cpu
            use_tpu_spawn = self.use_tpu and self._distrib_type == DistributedType.TPU_SPAWN
            use_ddp_cpu_torch_elastic = use_ddp_cpu_spawn and TorchElasticEnvironment.is_using_torchelastic()
            use_ddp_cpu_kubeflow = use_ddp_cpu_spawn and KubeflowEnvironment.is_using_kubeflow()
            use_ddp_cpu_slurm = use_ddp_cpu_spawn and self.is_slurm_managing_tasks
            use_ddp_sharded = self._distrib_type == DistributedType.DDP_SHARDED
            use_ddp_sharded_spawn = self._distrib_type == DistributedType.DDP_SHARDED_SPAWN
            use_ddp_fully_sharded = self._distrib_type == DistributedType.DDP_FULLY_SHARDED

            if use_tpu_spawn:
                ddp_plugin_cls = TPUSpawnPlugin
            elif use_ddp_sharded:
                ddp_plugin_cls = DDPShardedPlugin
            elif use_ddp_sharded_spawn:
                ddp_plugin_cls = DDPSpawnShardedPlugin
            elif (
                use_ddp_cpu_slurm
                or use_slurm_ddp
                or use_ddp_cpu_torch_elastic
                or use_torchelastic_ddp
                or use_kubeflow_ddp
                or use_ddp_cpu_kubeflow
            ):
                ddp_plugin_cls = DDPPlugin
            elif use_ddp_spawn or use_ddp_cpu_spawn:
                ddp_plugin_cls = DDPSpawnPlugin
            elif use_ddp_fully_sharded:
                ddp_plugin_cls = DDPFullyShardedPlugin
            else:
                ddp_plugin_cls = DDPPlugin

            plugin = ddp_plugin_cls(
                parallel_devices=self.parallel_devices, cluster_environment=self.cluster_environment
            )
        elif self.use_dp:
            plugin = DataParallelPlugin(parallel_devices=self.parallel_devices)
        elif self.use_horovod:
            plugin = HorovodPlugin(parallel_devices=self.parallel_devices)
        elif self.use_tpu and isinstance(self.tpu_cores, list):
            plugin = SingleTPUPlugin(self.tpu_id)
        elif self.use_ipu:
            plugin = IPUPlugin(parallel_devices=self.parallel_devices)
        else:
            single_gpu_ordinal = device_parser.determine_root_gpu_device(self.parallel_device_ids)
            plugin = SingleDevicePlugin(device=torch.device(f"cuda:{single_gpu_ordinal}" if self.use_gpu else "cpu"))
        return plugin

    def resolve_training_type_plugin(self, training_type: TrainingTypePlugin) -> TrainingTypePlugin:
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
        # as precision_plugin is dependent on training_type_plugin, make sure
        # that we first select training_type_plugin, then precision_plugin
        accelerator = acc_cls(training_type_plugin=self.training_type_plugin, precision_plugin=self.precision_plugin)
        # transfer ownership of the plugins to the accelerator
        self._training_type_plugin = proxy(self.training_type_plugin)
        self._precision_plugin = proxy(self.precision_plugin)

        return accelerator

    def select_cluster_environment(self) -> ClusterEnvironment:
        if self._cluster_environment is not None:
            return self._cluster_environment
        if self.is_slurm_managing_tasks:
            env = SLURMEnvironment()
        elif TorchElasticEnvironment.is_using_torchelastic():
            env = TorchElasticEnvironment()
        elif KubeflowEnvironment.is_using_kubeflow():
            env = KubeflowEnvironment()
        elif LSFEnvironment.is_using_lsf():
            env = LSFEnvironment()
        else:
            env = LightningEnvironment()
        return env

    def set_distributed_mode(self, distributed_backend: Optional[str] = None):

        if distributed_backend is None and self.is_training_type_in_plugins:
            return

        if distributed_backend is not None and distributed_backend in TrainingTypePluginsRegistry:
            self.distributed_backend = TrainingTypePluginsRegistry[distributed_backend]["distributed_backend"]
        elif distributed_backend is not None:
            self.distributed_backend = distributed_backend

        if isinstance(self.distributed_backend, Accelerator):
            return

        is_cpu_accelerator_type = self._accelerator_type and self._accelerator_type == DeviceType.CPU
        _use_cpu = is_cpu_accelerator_type or self.distributed_backend and "cpu" in self.distributed_backend

        if self.distributed_backend is None:
            if self.has_horovodrun():
                self._set_horovod_backend()
            elif self.num_gpus == 0 and self.num_nodes > 1:
                self._distrib_type = DistributedType.DDP
            elif self.num_gpus == 0 and self.num_processes > 1:
                self.distributed_backend = DistributedType.DDP_SPAWN
            elif self.num_gpus > 1 and not _use_cpu:
                rank_zero_warn(
                    "You requested multiple GPUs but did not specify a backend, e.g."
                    ' `Trainer(accelerator="dp"|"ddp"|"ddp2")`. Setting `accelerator="ddp_spawn"` for you.'
                )
                self.distributed_backend = DistributedType.DDP_SPAWN

        # special case with DDP on CPUs
        if self.distributed_backend == "ddp_cpu":
            if _TPU_AVAILABLE:
                raise MisconfigurationException(
                    "`accelerator='ddp_cpu'` is not supported on TPU machines. "
                    "Learn more: https://github.com/PyTorchLightning/pytorch-lightning/issues/7810"
                )
            self._distrib_type = DistributedType.DDP_SPAWN
            if self.num_gpus > 0:
                rank_zero_warn(
                    "You requested one or more GPUs, but set the backend to `ddp_cpu`. Training will not use GPUs."
                )
                self.parallel_device_ids = None
            if self.num_processes is None:
                # define the max CPU available
                self.num_processes = os.cpu_count()
        # special case with TPUs
        elif self.has_tpu and not _use_cpu:
            self._device_type = DeviceType.TPU
            if isinstance(self.tpu_cores, int):
                self._distrib_type = DistributedType.TPU_SPAWN
        elif self.has_ipu and not _use_cpu:
            self._device_type = DeviceType.IPU
        elif self.distributed_backend and self._distrib_type is None:
            self._distrib_type = DistributedType(self.distributed_backend)

        if self.num_gpus > 0 and not _use_cpu:
            self._device_type = DeviceType.GPU

        _gpu_distrib_types = (DistributedType.DP, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2)
        # DP and DDP2 cannot run without GPU
        if self.num_gpus == 0 and self._distrib_type in _gpu_distrib_types and not _use_cpu:

            if (self.num_nodes and self.num_nodes > 1) or (self.num_processes and self.num_processes > 1):
                if self._distrib_type in (DistributedType.DP, DistributedType.DDP2):
                    rank_zero_warn(
                        f"{self._distrib_type} is not supported on CPUs, hence setting the distributed type to `ddp`."
                    )
                    self._distrib_type = DistributedType.DDP
            else:
                rank_zero_warn("You are running on single node with no parallelization, so distributed has no effect.")
                self._distrib_type = None

        # finished configuring self._distrib_type, check ipython environment
        self.check_interactive_compatibility()

        # for DDP overwrite nb processes by requested GPUs
        if self._device_type == DeviceType.GPU and self._distrib_type in (
            DistributedType.DDP,
            DistributedType.DDP_SPAWN,
        ):
            self.num_processes = self.num_gpus

        if self._device_type == DeviceType.GPU and self._distrib_type == DistributedType.DDP2:
            self.num_processes = self.num_nodes

        # Horovod is an extra case...
        if self.distributed_backend == "horovod":
            self._set_horovod_backend()

        using_valid_distributed = self.use_ddp or self.use_ddp2
        if self.num_nodes > 1 and not using_valid_distributed:
            # throw error to force user to choose a supported distributed type such as ddp or ddp2
            raise MisconfigurationException(
                "Your chosen distributed type does not support num_nodes > 1. "
                "Please set accelerator=ddp or accelerator=ddp2."
            )

    def _set_horovod_backend(self):
        self.check_horovod()
        self._distrib_type = DistributedType.HOROVOD

        # Initialize Horovod to get rank / size info
        hvd.init()
        if self.has_gpu:
            # Horovod assigns one local GPU per process
            self.parallel_device_ids = list(range(hvd.local_size()))
        else:
            self.num_processes = hvd.local_size()

    def check_interactive_compatibility(self):
        """
        Raises a `MisconfigurationException` if the accelerator and/or plugin
        is not compatible with an interactive environment
        """
        from pytorch_lightning.utilities import _IS_INTERACTIVE

        if _IS_INTERACTIVE and self._distrib_type is not None and not self._distrib_type.is_interactive_compatible():
            raise MisconfigurationException(
                f"Selected distributed backend {self._distrib_type} is not compatible with an interactive"
                " environment. Run your code as a script, or choose one of the compatible backends:"
                f" {', '.join(DistributedType.interactive_compatible_types())}"
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
        # This allows the poptorch.Options that are passed into the IPUPlugin to be the source of truth,
        # which gives users the flexibility to not have to pass `ipus` flag directly to Trainer
        if isinstance(self._training_type_plugin, IPUPlugin) and self._device_type != DeviceType.IPU:
            self._device_type = DeviceType.IPU

    def configure_slurm_ddp(self):
        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.use_ddp or self.use_ddp2:
            num_requested_gpus = self.num_gpus * self.num_nodes
            num_slurm_tasks = 0
            try:
                num_slurm_tasks = int(os.environ["SLURM_NTASKS"])
                self.is_slurm_managing_tasks = num_slurm_tasks == num_requested_gpus

                # enable slurm cpu
                if num_requested_gpus == 0:
                    self.is_slurm_managing_tasks = num_slurm_tasks == self.num_processes

                # in interactive mode we don't manage tasks
                job_name = os.environ["SLURM_JOB_NAME"]
                if job_name == "bash":
                    self.is_slurm_managing_tasks = False

            except Exception:
                # likely not on slurm, so set the slurm managed flag to false
                self.is_slurm_managing_tasks = False

        # used for tests only, set this flag to simulate slurm managing a task
        try:
            should_fake = int(os.environ["FAKE_SLURM_MANAGING_TASKS"])
            if should_fake:
                self.is_slurm_managing_tasks = True
        except Exception:
            pass

        # notify user the that slurm is managing tasks
        if self.is_slurm_managing_tasks:
            rank_zero_info("Multi-processing is handled by Slurm.")
