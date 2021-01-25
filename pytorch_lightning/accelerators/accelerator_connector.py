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

from pytorch_lightning.utilities import HOROVOD_AVAILABLE
from pytorch_lightning import _logger as log
from pytorch_lightning import accelerators
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.cluster_environments.slurm_environment import SLURMEnvironment
from pytorch_lightning.cluster_environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.utilities import device_parser, rank_zero_only, TPU_AVAILABLE
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class AcceleratorConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.accelerator = None

    def on_trainer_init(
            self,
            num_processes,
            tpu_cores,
            accelerator,
            distributed_backend,
            auto_select_gpus,
            gpus,
            num_nodes,
            log_gpu_memory,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp,
            deterministic,
    ):
        # temp until we remove all dist backend references
        distributed_backend = self._map_deprecated_dist_backend(accelerator, distributed_backend)

        self.trainer.deterministic = deterministic

        torch.backends.cudnn.deterministic = self.trainer.deterministic
        if self.trainer.deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

        # distributed backend choice
        self.trainer.distributed_backend = distributed_backend.lower() if distributed_backend else None

        # init the default rank if exists
        # we need to call this here or NVIDIA flags and other messaging in init will show on all ranks
        # this way we only show it on rank 0
        if 'LOCAL_RANK' in os.environ:
            rank_zero_only.rank = int(os.environ['LOCAL_RANK'])

        # benchmarking
        self.trainer.benchmark = benchmark
        torch.backends.cudnn.benchmark = self.trainer.benchmark

        # Transfer params
        self.trainer.num_nodes = num_nodes
        self.trainer.log_gpu_memory = log_gpu_memory

        # sync-bn backend
        self.trainer.sync_batchnorm = sync_batchnorm

        self.trainer.tpu_cores = device_parser.parse_tpu_cores(tpu_cores)
        self.trainer.on_tpu = self.trainer.tpu_cores is not None

        self.trainer.tpu_id = self.trainer.tpu_cores[0] if isinstance(self.trainer.tpu_cores, list) else None

        if num_processes != 1 and distributed_backend != "ddp_cpu":
            rank_zero_warn("num_processes is only used for `accelerator='ddp_cpu'`. Ignoring it.")
        self.trainer.num_processes = num_processes

        # override with environment flag
        gpus = os.environ.get('PL_TRAINER_GPUS', gpus)
        self.trainer.gpus = gpus

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.trainer.gpus = self.trainer.tuner.pick_multiple_gpus(gpus)

        self.trainer.data_parallel_device_ids = device_parser.parse_gpu_ids(self.trainer.gpus)
        self.trainer.root_gpu = device_parser.determine_root_gpu_device(self.trainer.data_parallel_device_ids)
        self.trainer.root_device = torch.device("cpu")

        self.trainer.on_gpu = True if (self.trainer.data_parallel_device_ids and torch.cuda.is_available()) else False

        # tpu state flags
        self.trainer.use_tpu = False
        self.trainer.tpu_local_core_rank = None
        self.trainer.tpu_global_core_rank = None

        # distributed backend choice
        self.set_distributed_mode()

        # override dist backend when using tpus
        if self.trainer.on_tpu:
            self.trainer.distributed_backend = "tpu"
            self.trainer.use_tpu = True

        # init flags for SLURM+DDP to work
        self.trainer.world_size = 1
        self.trainer.interactive_ddp_procs = []

        # link up SLURM
        # TODO: this should be taken out of here... but depends too much on DDP
        self.trainer.slurm_connector.on_trainer_init(self.trainer.num_nodes)
        self.trainer.node_rank = self.determine_ddp_node_rank()
        self.trainer.local_rank = self.determine_local_rank()
        self.trainer.global_rank = 0

        # NVIDIA setup
        self.set_nvidia_flags(self.trainer.is_slurm_managing_tasks, self.trainer.data_parallel_device_ids)

        self.trainer.on_colab_kaggle = os.getenv('COLAB_GPU') or os.getenv('KAGGLE_URL_BASE')

        self.trainer.replace_sampler_ddp = replace_sampler_ddp

    def _map_deprecated_dist_backend(self, accelerator, distributed_backend):
        if distributed_backend is not None:
            rank_zero_warn(DeprecationWarning('distributed_backend has been renamed to accelerator. '
                                              'Deprecated in 1.0.0, will be removed in 1.2.0'))

        # temporary mapping until we remove all the distributed_backend references
        if accelerator is not None:
            self.accelerator = accelerator
            if isinstance(accelerator, Accelerator):
                self.accelerator.trainer = self
                distributed_backend = self.accelerator.nickname
            else:
                distributed_backend = accelerator
        return distributed_backend

    def _select_environment(self):
        if self.trainer.plugin_connector.cloud_environment:
            env = self.trainer.plugin_connector.cloud_environment
        elif self.trainer.is_slurm_managing_tasks:
            env = SLURMEnvironment()
        elif self._is_using_torchelastic():
            env = TorchElasticEnvironment()
        else:
            env = TorchElasticEnvironment()
        return env

    def _is_using_torchelastic(self):
        te_flags_passed = 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ)
        return te_flags_passed

    def select_accelerator(self):
        if self.trainer.accelerator_backend is not None:
            return self.trainer.accelerator_backend

        # ----------------------------------
        # Use the user provided accelerator
        # ----------------------------------
        # use the one the user passed in
        if self.accelerator is not None and isinstance(self.accelerator, Accelerator):
            self.accelerator.trainer = self.trainer
            self.accelerator.ddp_plugin = self.trainer.plugin_connector.ddp_plugin
            acc = self.accelerator
            return acc

        # ----------------------------------
        # choose an accelerator for the user
        # ----------------------------------
        use_slurm_ddp = self.trainer.use_ddp and self.trainer.is_slurm_managing_tasks

        # torchelastic or general non_slurm ddp
        te_flags_passed = 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ)
        use_torchelastic_ddp = self.trainer.use_ddp and te_flags_passed

        use_ddp_spawn = self.trainer.use_ddp and self.trainer.distributed_backend == "ddp_spawn"
        use_ddp_cpu_spawn = self.trainer.use_ddp and self.trainer.distributed_backend == "ddp_cpu"

        use_ddp_cpu_torch_elastic = use_ddp_cpu_spawn and self._is_using_torchelastic()
        use_ddp_cpu_slurm = use_ddp_cpu_spawn and self.trainer.is_slurm_managing_tasks

        # ddp script mode uses the same flags as TE
        # TODO: decouple from TE
        if os.environ.get('PL_IN_DDP_SUBPROCESS', False):
            use_torchelastic_ddp = False

        cluster_env = self._select_environment()

        # choose the appropriate accelerator backend
        if self.trainer.use_ddp2:
            accelerator_backend = accelerators.DDP2Accelerator(
                self.trainer,
                cluster_env,
                self.trainer.plugin_connector.ddp_plugin
            )

        elif use_ddp_cpu_slurm:
            accelerator_backend = accelerators.DDPCPUHPCAccelerator(
                self.trainer,
                cluster_env,
                self.trainer.plugin_connector.ddp_plugin
            )

        elif use_slurm_ddp:
            accelerator_backend = accelerators.DDPHPCAccelerator(
                self.trainer,
                cluster_env,
                self.trainer.plugin_connector.ddp_plugin
            )

        elif use_ddp_cpu_torch_elastic:
            accelerator_backend = accelerators.DDPCPUHPCAccelerator(
                self.trainer,
                cluster_env,
                self.trainer.plugin_connector.ddp_plugin
            )

        elif use_torchelastic_ddp:
            accelerator_backend = accelerators.DDPHPCAccelerator(
                self.trainer,
                cluster_env,
                self.trainer.plugin_connector.ddp_plugin
            )

        elif use_ddp_spawn:
            accelerator_backend = accelerators.DDPSpawnAccelerator(
                self.trainer,
                nprocs=self.trainer.num_processes,
                cluster_environment=cluster_env,
                ddp_plugin=self.trainer.plugin_connector.ddp_plugin
            )

        elif use_ddp_cpu_spawn:
            accelerator_backend = accelerators.DDPCPUSpawnAccelerator(
                self.trainer,
                nprocs=self.trainer.num_processes,
                cluster_environment=cluster_env,
                ddp_plugin=self.trainer.plugin_connector.ddp_plugin
            )

        elif self.trainer.distributed_backend == "ddp":
            accelerator_backend = accelerators.DDPAccelerator(
                self.trainer,
                cluster_env,
                ddp_plugin=self.trainer.plugin_connector.ddp_plugin
            )

        elif self.trainer.use_dp:
            accelerator_backend = accelerators.DataParallelAccelerator(self.trainer, cluster_env)

        elif self.trainer.use_horovod:
            accelerator_backend = accelerators.HorovodAccelerator(self.trainer, cluster_env)

        elif self.trainer.use_single_gpu:
            accelerator_backend = accelerators.GPUAccelerator(self.trainer, cluster_env)

        elif self.trainer.use_tpu:
            accelerator_backend = accelerators.TPUAccelerator(self.trainer, cluster_env)

        elif self.trainer.distributed_backend is None:
            accelerator_backend = accelerators.CPUAccelerator(self.trainer, cluster_env)
        else:
            raise MisconfigurationException(
                f'Trainer(accelerator={self.trainer.distributed_backend} is not a supported backend'
            )

        return accelerator_backend

    def set_distributed_mode(self):
        self.trainer.use_dp = False
        self.trainer.use_ddp = False
        self.trainer.use_ddp2 = False
        self.trainer.use_horovod = False
        self.trainer.use_single_gpu = False

        if self.trainer.distributed_backend is None:
            if self.has_horovodrun():
                self._set_horovod_backend()
            elif self.trainer.num_gpus == 0:
                if self.trainer.num_nodes > 1 or self.trainer.num_processes > 1:
                    self.trainer.use_ddp = True  # ddp_cpu
            elif self.trainer.num_gpus == 1:
                self.trainer.use_single_gpu = True
            elif self.trainer.num_gpus > 1:
                rank_zero_warn(
                    'You requested multiple GPUs but did not specify a backend, e.g.'
                    ' `Trainer(accelerator="dp"|"ddp"|"ddp2")`.'
                    ' Setting `accelerator="ddp_spawn"` for you.'
                )
                self.trainer.distributed_backend = "ddp_spawn"

        if self.trainer.distributed_backend == "dp":
            # do nothing if num_gpus == 0
            if self.trainer.num_gpus == 1:
                self.trainer.use_single_gpu = True
                self.trainer.use_dp = True
            elif self.trainer.num_gpus > 1:
                self.trainer.use_dp = True

        elif self.trainer.distributed_backend in ("ddp", "ddp_spawn"):
            if self.trainer.num_gpus == 0:
                if self.trainer.num_nodes > 1 or self.trainer.num_processes > 1:
                    self.trainer.use_ddp = True  # ddp_cpu
            elif self.trainer.num_gpus == 1:
                self.trainer.use_single_gpu = True
                self.trainer.use_ddp = True
            elif self.trainer.num_gpus > 1:
                self.trainer.use_ddp = True
                self.trainer.num_processes = self.trainer.num_gpus

        elif self.trainer.distributed_backend == "ddp2":
            # do nothing if num_gpus == 0
            if self.trainer.num_gpus >= 1:
                self.trainer.use_ddp2 = True
        elif self.trainer.distributed_backend == "ddp_cpu":
            if self.trainer.num_gpus > 0:
                rank_zero_warn(
                    'You requested one or more GPUs, but set the backend to `ddp_cpu`. Training will not use GPUs.'
                )
            self.trainer.use_ddp = True
            self.trainer.data_parallel_device_ids = None
            self.trainer.on_gpu = False
            self.trainer.on_cpu = True
        elif self.trainer.distributed_backend == "horovod":
            self._set_horovod_backend()

        # throw error to force user ddp or ddp2 choice
        if self.trainer.num_nodes > 1 and not (self.trainer.use_ddp2 or self.trainer.use_ddp):
            raise MisconfigurationException(
                'DataParallel does not support num_nodes > 1. '
                'To avoid this exception, set `accelerator="ddp"` or `accelerator="ddp2"`'
            )

        rank_zero_info(f'GPU available: {torch.cuda.is_available()}, used: {self.trainer.on_gpu}')
        num_cores = self.trainer.tpu_cores if self.trainer.tpu_cores is not None else 0
        rank_zero_info(f'TPU available: {TPU_AVAILABLE}, using: {num_cores} TPU cores')

        if torch.cuda.is_available() and not self.trainer.on_gpu:
            rank_zero_warn('GPU available but not used. Set the --gpus flag when calling the script.')

    def _set_horovod_backend(self):
        self.check_horovod()
        self.trainer.use_horovod = True

        # Initialize Horovod to get rank / size info
        hvd.init()
        if self.trainer.on_gpu:
            # Horovod assigns one local GPU per process
            self.trainer.root_gpu = hvd.local_rank()

    def check_horovod(self):
        """Raises a `MisconfigurationException` if the Trainer is not configured correctly for Horovod."""
        if not HOROVOD_AVAILABLE:
            raise MisconfigurationException(
                'Requested `accelerator="horovod"`, but Horovod is not installed.'
                'Install with \n $HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]'
            )

        if self.trainer.num_gpus > 1 or self.trainer.num_nodes > 1:
            raise MisconfigurationException(
                'Horovod does not support setting num_nodes / num_gpus explicitly. Use '
                'horovodrun / mpirun to configure the number of processes.'
            )

    @staticmethod
    def has_horovodrun():
        """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
        return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ

    def set_nvidia_flags(self, is_slurm_managing_tasks, data_parallel_device_ids):
        if data_parallel_device_ids is None:
            return

        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join([str(x) for x in range(torch.cuda.device_count())])
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        log.info(f'LOCAL_RANK: {self.trainer.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]')

    def determine_local_rank(self):
        if self.trainer.is_slurm_managing_tasks:
            return int(os.environ['SLURM_LOCALID'])
        return int(os.environ.get('LOCAL_RANK', 0))

    def determine_ddp_node_rank(self):
        if self.trainer.is_slurm_managing_tasks:
            return int(os.environ['SLURM_NODEID'])

        # torchelastic uses the envvar GROUP_RANK, whereas other systems(?) use NODE_RANK.
        # otherwise use given node rank or default to node rank 0
        env_vars = ['NODE_RANK', 'GROUP_RANK']
        node_ids = [(k, os.environ.get(k, None)) for k in env_vars]
        node_ids = [(k, v) for k, v in node_ids if v is not None]
        if len(node_ids) == 0:
            return 0
        if len(node_ids) > 1:
            log.warning(f"Multiple environment variables ({node_ids}) defined for node rank. Using the first one.")
        k, rank = node_ids.pop()
        rank_zero_info(f"Using environment variable {k} for node rank ({rank}).")
        return int(rank)
