from pytorch_lightning import accelerators
import os


class AcceleratorConnector:

    def __init__(self, trainer):
        self.trainer = trainer

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
