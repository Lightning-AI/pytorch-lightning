from pytorch_lightning import accelerators
import os
import torch
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning import _logger as log

try:
    import torch_xla
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class AcceleratorConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
            self,
            num_processes,
            tpu_cores,
            distributed_backend,
            auto_select_gpus,
            gpus,
            num_nodes,
            log_gpu_memory,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp,
            deterministic
    ):
        self.trainer.deterministic = deterministic
        torch.backends.cudnn.deterministic = self.trainer.deterministic
        if self.trainer.deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

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
            rank_zero_warn("num_processes is only used for distributed_backend=\"ddp_cpu\". Ignoring it.")
        self.trainer.num_processes = num_processes

        # override with environment flag
        gpus = os.environ.get('PL_TRAINER_GPUS', gpus)

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.trainer.gpus = self.trainer.tuner.pick_multiple_gpus(gpus)
        else:
            self.trainer.gpus = gpus

        self.trainer.data_parallel_device_ids = device_parser.parse_gpu_ids(self.trainer.gpus)
        self.trainer.root_gpu = device_parser.determine_root_gpu_device(self.trainer.data_parallel_device_ids)
        self.trainer.root_device = torch.device("cpu")

        self.trainer.on_gpu = True if (self.trainer.data_parallel_device_ids and torch.cuda.is_available()) else False

        # tpu state flags
        self.trainer.use_tpu = False
        self.trainer.tpu_local_core_rank = None
        self.trainer.tpu_global_core_rank = None

        # distributed backend choice
        self.trainer.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend)

        # override dist backend when using tpus
        if self.trainer.on_tpu:
            self.trainer.distributed_backend = 'tpu'
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

    def select_accelerator(self):
        # SLURM ddp
        use_slurm_ddp = self.trainer.use_ddp and self.trainer.is_slurm_managing_tasks

        # torchelastic or general non_slurm ddp
        te_flags_passed = 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ)
        use_torchelastic_ddp = self.trainer.use_ddp and te_flags_passed

        use_ddp_spawn = self.trainer.use_ddp and self.trainer.distributed_backend == 'ddp_spawn'
        use_ddp_cpu_spawn = self.trainer.use_ddp and self.trainer.distributed_backend == 'ddp_cpu'

        # choose the appropriate accelerator backend
        if self.trainer.use_ddp2:
            accelerator_backend = accelerators.DDP2Backend(self.trainer)

        elif use_slurm_ddp:
            accelerator_backend = accelerators.DDPBackend(self.trainer, mode='slurm_ddp')

        elif use_torchelastic_ddp:
            accelerator_backend = accelerators.DDPBackend(self.trainer, mode='torchelastic_ddp')

        elif use_ddp_spawn:
            accelerator_backend = accelerators.DDPSpawnBackend(self.trainer, nprocs=self.trainer.num_processes)

        elif use_ddp_cpu_spawn:
            accelerator_backend = accelerators.DDPCPUSpawnBackend(self.trainer, nprocs=self.trainer.num_processes)

        elif self.trainer.distributed_backend == 'ddp':
            accelerator_backend = accelerators.DDPBackend(self.trainer, mode='ddp')

        elif self.trainer.use_dp:
            accelerator_backend = accelerators.DataParallelBackend(self.trainer)

        elif self.trainer.use_horovod:
            accelerator_backend = accelerators.HorovodBackend(self.trainer)

        elif self.trainer.use_single_gpu:
            accelerator_backend = accelerators.GPUBackend(self.trainer)

        elif self.trainer.use_tpu:
            accelerator_backend = accelerators.TPUBackend(self.trainer)

        else:
            accelerator_backend = accelerators.CPUBackend(self.trainer)

        return accelerator_backend

    def set_distributed_mode(self, distributed_backend):
        self.trainer.use_dp = False
        self.trainer.use_ddp = False
        self.trainer.use_ddp2 = False
        self.trainer.use_horovod = False
        self.trainer.use_single_gpu = False

        if distributed_backend is None:
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
                    ' Trainer(distributed_backend=dp) (or ddp, ddp2).'
                    ' Setting distributed_backend=ddp_spawn for you.'
                )
                self.trainer.distributed_backend = 'ddp_spawn'
                distributed_backend = 'ddp_spawn'

        if distributed_backend == "dp":
            # do nothing if num_gpus == 0
            if self.trainer.num_gpus == 1:
                self.trainer.use_single_gpu = True
                self.trainer.use_dp = True
            elif self.trainer.num_gpus > 1:
                self.trainer.use_dp = True

        elif distributed_backend in ['ddp', 'ddp_spawn']:
            if self.trainer.num_gpus == 0:
                if self.trainer.num_nodes > 1 or self.trainer.num_processes > 1:
                    self.trainer.use_ddp = True  # ddp_cpu
            elif self.trainer.num_gpus == 1:
                self.trainer.use_single_gpu = True
                self.trainer.use_ddp = True
            elif self.trainer.num_gpus > 1:
                self.trainer.use_ddp = True
                self.trainer.num_processes = self.trainer.num_gpus

        elif distributed_backend == "ddp2":
            # do nothing if num_gpus == 0
            if self.trainer.num_gpus >= 1:
                self.trainer.use_ddp2 = True
        elif distributed_backend == "ddp_cpu":
            if self.trainer.num_gpus > 0:
                rank_zero_warn(
                    'You requested one or more GPUs, but set the backend to `ddp_cpu`. Training will not use GPUs.'
                )
            self.trainer.use_ddp = True
            self.trainer.data_parallel_device_ids = None
            self.trainer.on_gpu = False
        elif distributed_backend == 'horovod':
            self._set_horovod_backend()

        # throw error to force user ddp or ddp2 choice
        if self.trainer.num_nodes > 1 and not (self.trainer.use_ddp2 or self.trainer.use_ddp):
            raise MisconfigurationException(
                'DataParallel does not support num_nodes > 1. Switching to DistributedDataParallel for you. '
                'To silence this warning set distributed_backend=ddp or distributed_backend=ddp2'
            )

        rank_zero_info(f'GPU available: {torch.cuda.is_available()}, used: {self.trainer.on_gpu}')
        num_cores = self.trainer.tpu_cores if self.trainer.tpu_cores is not None else 0
        rank_zero_info(f'TPU available: {XLA_AVAILABLE}, using: {num_cores} TPU cores')

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
                'Requested `distributed_backend="horovod"`, but Horovod is not installed.'
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

        # when slurm is managing the task it sets the visible devices
        if not is_slurm_managing_tasks and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            if isinstance(data_parallel_device_ids, int):
                id_str = ','.join(str(x) for x in list(range(data_parallel_device_ids)))
                os.environ["CUDA_VISIBLE_DEVICES"] = id_str
            else:
                gpu_str = ','.join([str(x) for x in data_parallel_device_ids])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        # don't make this debug... this is good UX
        rank_zero_info(f'CUDA_VISIBLE_DEVICES: [{os.environ["CUDA_VISIBLE_DEVICES"]}]')

    def determine_local_rank(self):
        if self.trainer.is_slurm_managing_tasks:
            return int(os.environ['SLURM_LOCALID'])
        else:
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
