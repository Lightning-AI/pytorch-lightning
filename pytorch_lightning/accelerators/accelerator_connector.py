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
    DDPSpawnPlugin,
    HorovodPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    SingleDevicePlugin,
    SingleTPUPlugin,
    TPUHalfPrecisionPlugin,
    TPUSpawnPlugin,
)
from pytorch_lightning.plugins.environments import SLURMEnvironment, TorchElasticEnvironment
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


class BackendConnector(object):

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
        cluster_environment,
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
        self.cluster_environment = cluster_environment
        self.is_slurm_managing_tasks = False

        # init the default rank if exists
        # we need to call this here or NVIDIA flags and other messaging in init will show on all ranks
        # this way we only show it on rank 0
        if "LOCAL_RANK" in os.environ:
            rank_zero_only.rank = int(os.environ["LOCAL_RANK"])

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.gpus = pick_multiple_gpus(gpus)

        self.parallel_device_ids = device_parser.parse_gpu_ids(self.gpus)
        self.root_gpu = device_parser.determine_root_gpu_device(self.parallel_device_ids)

        self.set_distributed_mode()
        self.configure_slurm_ddp()

        self.accelerator = self.select_accelerator()

        # override dist backend when using tpus
        if self.on_tpu:
            self.distributed_backend = "tpu"
            self.use_tpu = True

        # init flags for SLURM+DDP to work
        self.world_size = 1
        self.interactive_ddp_procs = []
        self.global_rank = 0

        # NVIDIA setup
        # self.set_nvidia_flags(self.trainer.is_slurm_managing_tasks, self.trainer.data_parallel_device_ids)

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

        # TODO: move this to TPU accelerator/plugin
        self.on_colab_kaggle = os.getenv("COLAB_GPU") or os.getenv("KAGGLE_URL_BASE")

        self.replace_sampler_ddp = replace_sampler_ddp

    @property
    def on_cpu(self):
        return self._device_type == DeviceType.CPU

    @property
    def on_tpu(self):
        return self.tpu_cores is not None

    @property
    def tpu_id(self):
        if self.on_tpu:
            return self.tpu_cores[0]

        return None

    @property
    def on_gpu(self):
        gpus = self.parallel_device_ids
        return gpus is not None and len(gpus) > 0 and torch.cuda.is_available()

    @property
    def use_dp(self):
        return self._distrib_type == DistributedType.DP

    @property
    def use_ddp(self):
        return self._distrib_type in (DistributedType.DDP, DistributedType.DDP_SPAWN)

    @property
    def use_ddp2(self):
        return self._distrib_type == DistributedType.DDP2

    @property
    def use_horovod(self):
        return self._distrib_type == DistributedType.HOROVOD

    @property
    def num_gpus(self) -> int:
        gpus = self.parallel_device_ids
        if gpus is None:
            return 0
        return len(gpus)

    @property
    def parallel_devices(self):
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
    def is_using_torchelastic(self):
        te_flags_passed = "WORLD_SIZE" in os.environ and ("GROUP_RANK" in os.environ or "NODE_RANK" in os.environ)
        return te_flags_passed

    def select_precision_plugin(self):
        if self.precision == 32:
            self.amp_type = None
            return PrecisionPlugin()

        elif self.precision == 16:
            if self.on_tpu:
                return TPUHalfPrecisionPlugin()

            if self.amp_type == "native":
                if not _NATIVE_AMP_AVAILABLE:
                    rank_zero_warn(
                        "You have asked for native AMP but your PyTorch version does not support it."
                        " Consider upgrading with `pip install torch>=1.6`."
                        " We will attempt to use NVIDIA Apex for this session."
                    )
                    self.amp_type = "apex"
                else:
                    log.info("Using native 16bit precision.")
                    if self.distributed_backend == "ddp_sharded" or self.distributed_backend == "ddp_sharded_spawn":
                        return ShardedNativeMixedPrecisionPlugin()
                    self.amp_type = AMPType.NATIVE
                    return NativeMixedPrecisionPlugin()

            if self.amp_type == "apex":
                if not _APEX_AVAILABLE:
                    rank_zero_warn(
                        "You have asked for Apex AMP but you have not installed it yet."
                        " Install apex first using this guide: https://github.com/NVIDIA/apex#linux"
                    )
                else:
                    if self.distributed_backend == "ddp_sharded" or self.distributed_backend == "ddp_sharded_spawn":
                        raise MisconfigurationException(
                            "Sharded Plugin is not supported with Apex AMP, "
                            "please using native AMP for 16-bit precision."
                        )
                    log.info("Using APEX 16bit precision.")
                    self.amp_type = AMPType.APEX
                    return ApexMixedPrecisionPlugin(self.amp_level)
        else:
            raise NotImplementedError("We only support precisions 32 and 16!")

    def select_training_type_plugin(self):
        cluster_environment = self.select_cluster_environment()
        if self.use_ddp2:
            plugin = DDP2Plugin(parallel_devices=self.parallel_devices, cluster_environment=cluster_environment)
        elif self.use_ddp:
            use_slurm_ddp = self.use_ddp and self.is_slurm_managing_tasks
            use_torchelastic_ddp = self.use_ddp and self.is_using_torchelastic
            use_ddp_spawn = self._distrib_type == DistributedType.DDP_SPAWN
            use_ddp_cpu_spawn = self.use_ddp and self.on_cpu
            use_ddp_cpu_torch_elastic = use_ddp_cpu_spawn and self.is_using_torchelastic
            use_ddp_cpu_slurm = use_ddp_cpu_spawn and self.is_slurm_managing_tasks
            # use_ddp_sharded = self.distributed_backend == "ddp_sharded"
            # use_ddp_sharded_spawn = self.distributed_backend == "ddp_sharded_spawn"

            if self.on_tpu:
                ddp_plugin_cls = TPUSpawnPlugin

            # ddp script mode uses the same flags as TE
            # TODO: decouple from TE
            if os.environ.get("PL_IN_DDP_SUBPROCESS", False):
                use_torchelastic_ddp = False

            # fixme
            # if use_ddp_sharded:
            #     ddp_plugin_cls = DDPShardedPlugin
            # elif use_ddp_sharded_spawn:
            #     ddp_plugin_cls = DDPSpawnShardedPlugin
            if use_ddp_cpu_slurm or use_slurm_ddp or use_ddp_cpu_torch_elastic or use_torchelastic_ddp:
                ddp_plugin_cls = DDPPlugin
            elif use_ddp_spawn or use_ddp_cpu_spawn:
                ddp_plugin_cls = DDPSpawnPlugin
            else:
                ddp_plugin_cls = DDPPlugin

            plugin = ddp_plugin_cls(
                parallel_devices=self.parallel_devices,
                num_nodes=self.num_nodes,
                cluster_environment=cluster_environment,
                sync_batchnorm=self.sync_batchnorm,
            )
        elif self.use_dp:
            plugin = DataParallelPlugin(parallel_devices=self.parallel_devices)
        elif self.use_horovod:
            plugin = HorovodPlugin(parallel_devices=self.parallel_devices)
        elif self.on_tpu:
            plugin = SingleTPUPlugin(self.tpu_id)
        else:
            plugin = SingleDevicePlugin(device=torch.device(f"cuda:{self.root_gpu}" if self.on_gpu else "cpu"))
        return plugin

    def select_accelerator(self):
        if isinstance(self.distributed_backend, Accelerator):
            # custom accelerator from user
            return self.distributed_backend

        if self.on_gpu:
            acc_cls = GPUAccelerator
        elif self.on_tpu:
            acc_cls = TPUAccelerator
        else:
            acc_cls = CPUAccelerator

        return acc_cls(
            precision_plugin=self.select_precision_plugin(),
            training_type_plugin=self.select_training_type_plugin(),
        )

    def select_cluster_environment(self):
        if self.cluster_environment is not None:
            return self.cluster_environment
        if self.is_slurm_managing_tasks:
            env = SLURMEnvironment()
        elif self.is_using_torchelastic:
            env = TorchElasticEnvironment()
            # TODO: decouple DDP from TE
            #   maybe introduce a DefaultEnvironment?
            os.environ["PL_IN_DDP_SUBPROCESS"] = "1"
        else:
            # TODO: maybe introduce a DefaultEnvironment?
            env = TorchElasticEnvironment()
        return env

    def set_distributed_mode(self):

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
            self.data_parallel_device_ids = None
            if self.num_gpus > 0:
                rank_zero_warn(
                    'You requested one or more GPUs, but set the backend to `ddp_cpu`. Training will not use GPUs.'
                )
            if self.num_processes is None:
                # define the max CPU available
                self.num_processes = os.cpu_count()
        # special case with TPUs
        elif self.distributed_backend == 'tpu':
            self._device_type = DeviceType.TPU
        # set all other requested distrib. types adn if it was not set in the
        elif self.distributed_backend and self._distrib_type is None:
            self._distrib_type = DistributedType(self.distributed_backend)

        # unless you request explicitly for CPU and some GPU are available use them
        _on_cpu = self.distributed_backend and 'cpu' in self.distributed_backend
        if (self.num_gpus > 0 and not _on_cpu):
            self._device_type = DeviceType.GPU

        _distrib_types = (DistributedType.DP, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2)
        # DP and DDP2 cannot run without GPU
        if (self.num_gpus == 0 and self._distrib_type in _distrib_types):
            rank_zero_warn(
                'You requested distributed training on GPUs, but none is available, so we set backend to `ddp_cpu`.'
            )
            # todo: in some cases it yield in comarison None and int
            if ((self.num_nodes and self.num_nodes > 1) or (self.num_processes and self.num_processes > 1)):
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
            self.root_gpu = hvd.local_rank()
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
    def has_horovodrun():
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
