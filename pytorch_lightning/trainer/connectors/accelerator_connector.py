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

import os
from typing import List, Optional, Sequence, Union

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from pytorch_lightning.plugins import (
    ApexMixedPrecisionPlugin,
    DataParallelPlugin,
    DDP2Plugin,
    DDPPlugin,
    DDPShardedPlugin,
    DDPSpawnPlugin,
    DDPSpawnShardedPlugin,
    DeepSpeedPlugin,
    DeepSpeedPrecisionPlugin,
    HorovodPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    SingleDevicePlugin,
    SingleTPUPlugin,
    TPUHalfPrecisionPlugin,
    TPUSpawnPlugin,
    TrainingTypePlugin,
)
from pytorch_lightning.plugins.environments import ClusterEnvironment, SLURMEnvironment, TorchElasticEnvironment
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.utilities import (
    _APEX_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _NATIVE_AMP_AVAILABLE,
    _TPU_AVAILABLE,
    AMPType,
    device_parser,
    DeviceType,
    DistributedType,
    rank_zero_only,
)
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class AcceleratorConnector(object):

    def __init__(
        self,
        num_processes,
        tpu_cores,
        distributed_backend,
        auto_select_gpus,
        gpus,
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

        self.num_processes = num_processes
        self.tpu_cores = device_parser.parse_tpu_cores(tpu_cores)
        self.distributed_backend = distributed_backend
        self.auto_select_gpus = auto_select_gpus
        self.gpus = gpus
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

        # init the default rank if exists
        # we need to call this here or NVIDIA flags and other messaging in init will show on all ranks
        # this way we only show it on rank 0
        if "LOCAL_RANK" in os.environ:
            rank_zero_only.rank = int(os.environ["LOCAL_RANK"])

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.gpus = pick_multiple_gpus(gpus)

        self.parallel_device_ids = device_parser.parse_gpu_ids(self.gpus)

        self.set_distributed_mode()
        self.configure_slurm_ddp()

        self.handle_given_plugins(plugins)

        self.accelerator = self.select_accelerator()

        # override dist backend when using tpus
        if self.on_tpu:
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

    def handle_given_plugins(
        self, plugins: Optional[Union[ClusterEnvironment, TrainingTypePlugin, PrecisionPlugin, Sequence]]
    ):
        plugins = plugins if plugins is not None else []

        if isinstance(plugins, str):
            plugins = [plugins]

        if not isinstance(plugins, Sequence):
            plugins = [plugins]

        training_type = None
        precision = None
        cluster_environment = None

        for plug in plugins:
            if isinstance(plug, str):
                self.set_distributed_mode(plug)

            elif isinstance(plug, TrainingTypePlugin):
                if training_type is None:
                    training_type = plug

                else:
                    raise MisconfigurationException(
                        'You can only specify one precision and one training type plugin.'
                        f' Found more than 1 training type plugin: {type(plug).__name__}'
                    )
            elif isinstance(plug, PrecisionPlugin):
                if precision is None:
                    precision = plug
                else:
                    raise MisconfigurationException(
                        'You can only specify one precision and one training type plugin.'
                        f' Found more than 1 precision plugin: {type(plug).__name__}'
                    )

            elif isinstance(plug, ClusterEnvironment):
                if cluster_environment is None:
                    cluster_environment = plug
                else:
                    raise MisconfigurationException(
                        'You can only specify one cluster environment. Found more than 1 cluster environment plugin'
                    )
            else:
                raise MisconfigurationException(
                    f'Found invalid type for plugin {plug}. Expected a precision or training type plugin.'
                )

        self._training_type_plugin = training_type
        self._training_type_plugin = self.training_type_plugin
        self._precision_plugin = precision
        self._cluster_environment = cluster_environment or self.select_cluster_environment()

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        if self._precision_plugin is None:
            self._precision_plugin = self.select_precision_plugin()
        return self._precision_plugin

    @property
    def training_type_plugin(self) -> TrainingTypePlugin:
        if self._training_type_plugin is None:
            self._training_type_plugin = self.select_training_type_plugin()
        else:
            self._training_type_plugin = self.resolve_training_type_plugin(self._training_type_plugin)

        return self._training_type_plugin

    @property
    def cluster_environment(self) -> ClusterEnvironment:
        return self._cluster_environment

    @property
    def on_cpu(self) -> bool:
        return self._device_type == DeviceType.CPU

    @property
    def on_tpu(self) -> bool:
        return self.tpu_cores is not None

    @property
    def tpu_id(self) -> Optional[int]:
        if self.on_tpu and isinstance(self.tpu_cores, list):
            return self.tpu_cores[0]

        return None

    @property
    def on_gpu(self) -> bool:
        gpus = self.parallel_device_ids
        return gpus is not None and len(gpus) > 0 and torch.cuda.is_available()

    @property
    def use_dp(self) -> bool:
        return self._distrib_type == DistributedType.DP

    @property
    def use_ddp(self) -> bool:
        return self._distrib_type in (
            DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP_SHARDED,
            DistributedType.DDP_SHARDED_SPAWN, DistributedType.DEEPSPEED
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
    def is_distributed(self) -> bool:
        is_distributed = self.use_ddp or self.use_ddp2 or self.use_horovod
        if self.on_tpu:
            is_distributed |= self.training_type_plugin.is_distributed
        return is_distributed

    @property
    def num_gpus(self) -> int:
        gpus = self.parallel_device_ids
        if gpus is None:
            return 0
        return len(gpus)

    @property
    def parallel_devices(self) -> Union[List[torch.device], int]:
        if self.on_gpu:
            devices = [torch.device("cuda", i) for i in self.parallel_device_ids]
        elif self.on_tpu:
            # explicitly don't make a tpu device here!
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/3169
            devices = [i for i in self.parallel_device_ids]
        else:
            devices = [torch.device("cpu")] * self.num_processes
        return devices

    @property
    def root_gpu(self) -> Optional[int]:
        return self.accelerator.root_device.index if not isinstance(self.accelerator, TPUAccelerator) else None

    @property
    def is_using_torchelastic(self) -> bool:
        te_flags_passed = "WORLD_SIZE" in os.environ and ("GROUP_RANK" in os.environ or "NODE_RANK" in os.environ)
        return te_flags_passed

    def select_precision_plugin(self) -> PrecisionPlugin:
        # set precision type
        self.amp_type = AMPType.from_str(self.amp_type)

        if self._distrib_type == DistributedType.DEEPSPEED or isinstance(self._training_type_plugin, DeepSpeedPlugin):
            return DeepSpeedPrecisionPlugin(self.precision)

        if self.precision == 32:
            return PrecisionPlugin()

        elif self.precision == 16:
            if self.on_tpu:
                return TPUHalfPrecisionPlugin()

            if self.amp_type == AMPType.NATIVE:
                if self.on_cpu:
                    raise MisconfigurationException(
                        "You have asked for native AMP on CPU, but AMP is only available on GPU."
                    )
                elif not _NATIVE_AMP_AVAILABLE:
                    msg = "You have asked for native AMP but your PyTorch version does not support it." \
                          " Consider upgrading with `pip install torch>=1.6`."
                    if _APEX_AVAILABLE:
                        self.amp_type = AMPType.APEX
                        msg += " We will attempt to use NVIDIA Apex for this session."
                        rank_zero_warn(msg)
                    else:
                        raise MisconfigurationException(msg)
                else:
                    log.info("Using native 16bit precision.")
                    if isinstance(self.training_type_plugin, (DDPShardedPlugin, DDPSpawnShardedPlugin)):
                        return ShardedNativeMixedPrecisionPlugin()
                    return NativeMixedPrecisionPlugin()

            if self.amp_type == AMPType.APEX:
                if not _APEX_AVAILABLE:
                    raise MisconfigurationException(
                        "You have asked for Apex AMP but you have not installed it yet."
                        " Install apex first using this guide: https://github.com/NVIDIA/apex#linux"
                    )
                if isinstance(self.training_type_plugin, (DDPShardedPlugin, DDPSpawnShardedPlugin)):
                    raise MisconfigurationException(
                        "Sharded Plugin is not supported with Apex AMP,"
                        " please using native AMP for 16-bit precision."
                    )
                log.info("Using APEX 16bit precision.")
                return ApexMixedPrecisionPlugin(self.amp_level)

        raise NotImplementedError("We only support precisions 32 and 16!")

    def select_training_type_plugin(self) -> TrainingTypePlugin:
        if self.use_ddp2:
            plugin = DDP2Plugin(parallel_devices=self.parallel_devices, cluster_environment=self.cluster_environment)
        elif self.use_ddp and self.use_deepspeed:
            plugin = DeepSpeedPlugin(
                num_nodes=self.num_nodes,
                cluster_environment=self.select_cluster_environment(),
                parallel_devices=self.parallel_devices
            )
        elif self.use_ddp:
            use_slurm_ddp = self.use_ddp and self.is_slurm_managing_tasks
            use_torchelastic_ddp = self.use_ddp and self.is_using_torchelastic
            use_ddp_spawn = self._distrib_type == DistributedType.DDP_SPAWN
            use_ddp_cpu_spawn = self.use_ddp and self.on_cpu
            use_ddp_cpu_torch_elastic = use_ddp_cpu_spawn and self.is_using_torchelastic
            use_ddp_cpu_slurm = use_ddp_cpu_spawn and self.is_slurm_managing_tasks
            use_ddp_sharded = self._distrib_type == DistributedType.DDP_SHARDED
            use_ddp_sharded_spawn = self._distrib_type == DistributedType.DDP_SHARDED_SPAWN

            # TODO: decouple from TE
            # ddp script mode uses the same flags as TE
            if os.environ.get("PL_IN_DDP_SUBPROCESS", False):
                use_torchelastic_ddp = False

            if self.on_tpu:
                ddp_plugin_cls = TPUSpawnPlugin
            elif use_ddp_sharded:
                ddp_plugin_cls = DDPShardedPlugin
            elif use_ddp_sharded_spawn:
                ddp_plugin_cls = DDPSpawnShardedPlugin
            elif use_ddp_cpu_slurm or use_slurm_ddp or use_ddp_cpu_torch_elastic or use_torchelastic_ddp:
                ddp_plugin_cls = DDPPlugin
            elif use_ddp_spawn or use_ddp_cpu_spawn:
                ddp_plugin_cls = DDPSpawnPlugin
            else:
                ddp_plugin_cls = DDPPlugin

            plugin = ddp_plugin_cls(
                parallel_devices=self.parallel_devices,
                num_nodes=self.num_nodes,
                cluster_environment=self.cluster_environment,
                sync_batchnorm=self.sync_batchnorm,
            )
        elif self.use_dp:
            plugin = DataParallelPlugin(parallel_devices=self.parallel_devices)
        elif self.use_horovod:
            plugin = HorovodPlugin(parallel_devices=self.parallel_devices)
        elif self.on_tpu:
            if isinstance(self.tpu_cores, list):
                plugin = SingleTPUPlugin(self.tpu_id)
            else:
                plugin = TPUSpawnPlugin(parallel_devices=list(range(self.tpu_cores)))
        else:
            single_gpu_ordinal = device_parser.determine_root_gpu_device(self.parallel_device_ids)
            plugin = SingleDevicePlugin(device=torch.device(f"cuda:{single_gpu_ordinal}" if self.on_gpu else "cpu"))
        return plugin

    def resolve_training_type_plugin(self, training_type: TrainingTypePlugin) -> TrainingTypePlugin:
        # necessary for when the user has passed in a plugin
        if hasattr(training_type, 'parallel_devices') and not getattr(training_type, 'parallel_devices'):
            training_type.parallel_devices = self.parallel_devices
            if hasattr(training_type, 'num_processes'):
                training_type.num_processes = len(self.parallel_devices)

        if hasattr(training_type, 'cluster_environment') and getattr(training_type, 'cluster_environment') is None:
            training_type.cluster_environment = self.select_cluster_environment()

        if hasattr(training_type, 'num_nodes') and getattr(training_type, 'num_nodes') is None:
            training_type.num_nodes = self.num_nodes

        return training_type

    def select_accelerator(self) -> Accelerator:
        if isinstance(self.distributed_backend, Accelerator):
            # custom accelerator from user
            if self._precision_plugin is not None or self._training_type_plugin is not None:
                # plugins also specified by user
                rank_zero_warn(
                    'Specified `Precision` and `TrainingType` plugins will be ignored,'
                    ' since an `Accelerator` instance was provided.'
                )
            return self.distributed_backend

        if self.on_gpu:
            acc_cls = GPUAccelerator
        elif self.on_tpu:
            acc_cls = TPUAccelerator
        else:
            acc_cls = CPUAccelerator

        return acc_cls(
            precision_plugin=self.precision_plugin,
            training_type_plugin=self.training_type_plugin,
        )

    def select_cluster_environment(self) -> ClusterEnvironment:
        if self._cluster_environment is not None:
            return self._cluster_environment
        if self.is_slurm_managing_tasks:
            env = SLURMEnvironment()
            # TODO: decouple DDP from SLURM
            #   refactor and let generic cluster env hold the information about who spawns the processes
            os.environ["PL_IN_DDP_SUBPROCESS"] = "1"
        elif self.is_using_torchelastic:
            env = TorchElasticEnvironment()
            # TODO: decouple DDP from TE
            #   refactor and let generic cluster env hold the information about who spawns the processes
            os.environ["PL_IN_DDP_SUBPROCESS"] = "1"
        else:
            # TODO: maybe introduce a DefaultEnvironment?
            env = TorchElasticEnvironment()
        return env

    def set_distributed_mode(self, distributed_backend: Optional[str] = None):

        if distributed_backend is not None:
            self.distributed_backend = distributed_backend

        if isinstance(self.distributed_backend, Accelerator):
            return

        if self.distributed_backend is None:
            if self.has_horovodrun():
                self._set_horovod_backend()
            elif self.num_gpus == 0 and (self.num_nodes > 1 or self.num_processes > 1):
                self._distrib_type = DistributedType.DDP
            elif self.num_gpus > 1:
                rank_zero_warn(
                    'You requested multiple GPUs but did not specify a backend, e.g.'
                    ' `Trainer(accelerator="dp"|"ddp"|"ddp2")`. Setting `accelerator="ddp_spawn"` for you.'
                )
                self.distributed_backend = "ddp_spawn"

        # special case with DDP on CPUs
        if self.distributed_backend == "ddp_cpu":
            self._distrib_type = DistributedType.DDP
            if self.num_gpus > 0:
                rank_zero_warn(
                    'You requested one or more GPUs, but set the backend to `ddp_cpu`. Training will not use GPUs.'
                )
                self.parallel_device_ids = None
            if self.num_processes is None:
                # define the max CPU available
                self.num_processes = os.cpu_count()
        # special case with TPUs
        elif self.distributed_backend == 'tpu':
            self._device_type = DeviceType.TPU
        elif self.distributed_backend and self._distrib_type is None:
            self._distrib_type = DistributedType(self.distributed_backend)

        # unless you request explicitly for CPU and some GPU are available use them
        _on_cpu = self.distributed_backend and 'cpu' in self.distributed_backend
        if self.num_gpus > 0 and not _on_cpu:
            self._device_type = DeviceType.GPU

        _distrib_types = (DistributedType.DP, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2)
        # DP and DDP2 cannot run without GPU
        if self.num_gpus == 0 and self._distrib_type in _distrib_types and not _on_cpu:
            rank_zero_warn(
                'You requested distributed training on GPUs, but none is available, so we set backend to `ddp_cpu`.'
            )
            # todo: in some cases it yield in comarison None and int
            if (self.num_nodes and self.num_nodes > 1) or (self.num_processes and self.num_processes > 1):
                self._distrib_type = DistributedType.DDP
            else:
                rank_zero_warn('You are running on single node with no parallelization, so distributed has no effect.')
                self._distrib_type = None

        # for DDP overwrite nb processes by requested GPUs
        if (
            self._device_type == DeviceType.GPU
            and self._distrib_type in (DistributedType.DDP, DistributedType.DDP_SPAWN)
        ):
            self.num_processes = self.num_gpus

        if (self._device_type == DeviceType.GPU and self._distrib_type == DistributedType.DDP2):
            self.num_processes = self.num_nodes

        # Horovod is an extra case...
        if self.distributed_backend == "horovod":
            self._set_horovod_backend()

        # throw error to force user ddp or ddp2 choice
        _ddp = (DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2)
        if (self.num_nodes > 1 and self._distrib_type not in _ddp):
            raise MisconfigurationException(
                'DataParallel does not support num_nodes > 1. Switching to DistributedDataParallel for you. '
                'To silence this warning set `accelerator="ddp"` or `accelerator="ddp2"`'
            )

        rank_zero_info(f'GPU available: {torch.cuda.is_available()}, used: {self._device_type == DeviceType.GPU}')
        num_cores = self.tpu_cores if self.tpu_cores is not None else 0
        rank_zero_info(f'TPU available: {_TPU_AVAILABLE}, using: {num_cores} TPU cores')

        if torch.cuda.is_available() and self._device_type != DeviceType.GPU:
            rank_zero_warn("GPU available but not used. Set the --gpus flag when calling the script.")

    def _set_horovod_backend(self):
        self.check_horovod()
        self._distrib_type = DistributedType.HOROVOD

        # Initialize Horovod to get rank / size info
        hvd.init()
        if self.on_gpu:
            # Horovod assigns one local GPU per process
            self.parallel_device_ids = list(range(hvd.local_size()))
        else:
            self.num_processes = hvd.local_size()

    def check_horovod(self):
        """Raises a `MisconfigurationException` if the Trainer is not configured correctly for Horovod."""
        if not _HOROVOD_AVAILABLE:
            raise MisconfigurationException(
                'Requested `distributed_backend="horovod"`, but Horovod is not installed.'
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
        return "OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ

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
