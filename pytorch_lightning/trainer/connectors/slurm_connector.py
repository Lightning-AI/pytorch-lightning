import os
import re
import signal
from subprocess import call
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.distributed import rank_zero_info
import torch.distributed as torch_distrib
import torch


class SLURMConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, num_gpu_nodes):
        self.configure_slurm_ddp(num_gpu_nodes)

    def configure_slurm_ddp(self, num_gpu_nodes):
        self.trainer.is_slurm_managing_tasks = False

        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            self.trainer.num_requested_gpus = self.trainer.num_gpus * num_gpu_nodes
            self.trainer.num_slurm_tasks = 0
            try:
                self.trainer.num_slurm_tasks = int(os.environ['SLURM_NTASKS'])
                self.trainer.is_slurm_managing_tasks = self.trainer.num_slurm_tasks == self.trainer.num_requested_gpus

                # enable slurm cpu
                if self.trainer.num_requested_gpus == 0:
                    self.trainer.is_slurm_managing_tasks = self.trainer.num_slurm_tasks == self.trainer.num_processes

                # in interactive mode we don't manage tasks
                job_name = os.environ['SLURM_JOB_NAME']
                if job_name == 'bash':
                    self.trainer.is_slurm_managing_tasks = False

            except Exception:
                # likely not on slurm, so set the slurm managed flag to false
                self.trainer.is_slurm_managing_tasks = False

        # used for tests only, set this flag to simulate slurm managing a task
        try:
            should_fake = int(os.environ['FAKE_SLURM_MANAGING_TASKS'])
            if should_fake:
                self.trainer.is_slurm_managing_tasks = True
        except Exception:
            pass

        # notify user the that slurm is managing tasks
        if self.trainer.is_slurm_managing_tasks:
            rank_zero_info('Multi-processing is handled by Slurm.')

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name, numbers = root_node.split('[', maxsplit=1)
            number = numbers.split(',', maxsplit=1)[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def register_slurm_signal_handlers(self):
        # see if we're using slurm (not interactive)
        on_slurm = False
        try:
            job_name = os.environ['SLURM_JOB_NAME']
            if job_name != 'bash':
                on_slurm = True
        except Exception:
            pass

        if on_slurm:
            log.info('Set SLURM handle signals.')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        if self.trainer.is_global_zero:
            # save weights
            log.info('handling SIGUSR1')
            self.trainer.checkpoint_connector.hpc_save(self.trainer.weights_save_path, self.trainer.logger)

            # find job id
            job_id = os.environ['SLURM_JOB_ID']
            cmd = ['scontrol', 'requeue', job_id]

            # requeue job
            log.info(f'requeing job {job_id}...')
            result = call(cmd)

            # print result text
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

            # close experiment to avoid issues
            self.trainer.logger.close()

    def term_handler(self, signum, frame):
        # save
        log.info("bypassing sigterm")

    def connect_ddp(self, global_rank: int, world_size: int) -> None:
        """"""
        """
        Sets up environment variables necessary for pytorch distributed communications
        based on slurm environment.
        """
        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ["SLURM_JOB_ID"]
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ["MASTER_PORT"]
        except Exception:
            os.environ["MASTER_PORT"] = str(default_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        # figure out the root node addr
        try:
            root_node = os.environ["SLURM_NODELIST"].split(" ")[0]
        except Exception:
            root_node = "127.0.0.1"

        root_node = self.trainer.slurm_connector.resolve_root_node_address(root_node)
        os.environ["MASTER_ADDR"] = root_node
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")

        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(
                f"initializing ddp (SLURM): GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
            )
            torch_distrib.init_process_group(
                torch_backend, rank=global_rank, world_size=world_size
            )
