from pytorch_lightning import accelerators
import os
import torch
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only


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
        self.trainer.set_distributed_mode(distributed_backend)

        # override dist backend when using tpus
        if self.trainer.on_tpu:
            self.trainer.distributed_backend = 'tpu'
            self.trainer.init_tpu()

        # init flags for SLURM+DDP to work
        self.trainer.world_size = 1
        self.trainer.interactive_ddp_procs = []
        self.trainer.configure_slurm_ddp(self.trainer.num_nodes)
        self.trainer.node_rank = self.trainer.determine_ddp_node_rank()
        self.trainer.local_rank = self.trainer.determine_local_rank()
        self.trainer.global_rank = 0

        # NVIDIA setup
        self.trainer.set_nvidia_flags(self.trainer.is_slurm_managing_tasks, self.trainer.data_parallel_device_ids)

        self.trainer.on_colab_kaggle = os.getenv('COLAB_GPU') or os.getenv('KAGGLE_URL_BASE')

        self.trainer.replace_sampler_ddp = replace_sampler_ddp

    def select_accelerator(self):
        # SLURM ddp
        use_slurm_ddp = self.trainer.use_ddp and self.trainer.is_slurm_managing_tasks

        # torchelastic or general non_slurm ddp
        te_flags_passed = 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ)
        use_torchelastic_ddp = self.trainer.use_ddp and te_flags_passed

        use_ddp_spawn = self.trainer.use_ddp and self.trainer.distributed_backend in ['ddp_cpu', 'ddp_spawn']

        # choose the appropriate accelerator backend
        if self.trainer.use_ddp2:
            accelerator_backend = accelerators.DDP2Backend(self.trainer)

        elif use_slurm_ddp:
            accelerator_backend = accelerators.DDPBackend(self.trainer, mode='slurm_ddp')

        elif use_torchelastic_ddp:
            accelerator_backend = accelerators.DDPBackend(self.trainer, mode='torchelastic_ddp')

        elif use_ddp_spawn:
            accelerator_backend = accelerators.DDPSpawnBackend(self.trainer, nprocs=self.trainer.num_processes)

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
