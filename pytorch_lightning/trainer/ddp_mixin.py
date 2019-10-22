import os
import re
import warnings

import torch
import torch.distributed as dist

from pytorch_lightning.pt_overrides.override_data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerDDPMixin(object):
    def set_distributed_mode(self, distributed_backend, nb_gpu_nodes):
        # skip for CPU
        if self.num_gpus == 0:
            return

        # single GPU case
        # in single gpu case we allow ddp so we can train on multiple
        # nodes, 1 gpu per node
        if self.num_gpus == 1:
            self.single_gpu = True

            if distributed_backend is not None:
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
                self.use_ddp2 = distributed_backend == 'ddp2'

                # disable single gpu when using ddp2
                if self.use_ddp2:
                    self.single_gpu = False

        # multiple GPU case
        elif self.num_gpus > 1:
            if distributed_backend is not None:
                # DP, DDP case
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
                self.use_ddp2 = distributed_backend == 'ddp2'

            elif distributed_backend is None:
                m = 'When using multiple GPUs set ' \
                    'Trainer(distributed_backend=dp) (or ddp)'
                raise MisconfigurationException(m)

        # use ddp automatically if nb_gpu_nodes > 1
        if nb_gpu_nodes > 1 and self.use_dp:  # pragma: no cover
            self.use_ddp = True
            self.use_dp = False
            w = 'DataParallel does not support nb_gpu_nodes > 1. ' \
                'Switching to DistributedDataParallel for you. ' \
                'To silence this warning set distributed_backend=ddp'
            warnings.warn(w)

        print('gpu available: {}, used: {}'.format(torch.cuda.is_available(), self.on_gpu))

    def configure_slurm_ddp(self, nb_gpu_nodes):
        self.is_slurm_managing_tasks = False

        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.use_ddp:
            self.nb_requested_gpus = self.num_gpus * nb_gpu_nodes
            self.nb_slurm_tasks = 0
            try:
                self.nb_slurm_tasks = int(os.environ['SLURM_NTASKS'])
                self.is_slurm_managing_tasks = self.nb_slurm_tasks == self.nb_requested_gpus

                # in interactive mode we don't manage tasks
                job_name = os.environ['SLURM_JOB_NAME']
                if job_name == 'bash':
                    self.is_slurm_managing_tasks = False

            except Exception:
                # likely not on slurm, so set the slurm managed flag to false
                self.is_slurm_managing_tasks = False

        # used for tests only, set this flag to simulate slurm managing a task
        try:
            should_fake = int(os.environ['FAKE_SLURM_MANAGING_TASKS'])
            if should_fake:
                self.is_slurm_managing_tasks = True
        except Exception as e:
            pass

    def set_nvidia_flags(self, is_slurm_managing_tasks, data_parallel_device_ids):
        if data_parallel_device_ids is None:
            return

        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # when slurm is managing the task it sets the visible devices
        if not is_slurm_managing_tasks:
            if type(data_parallel_device_ids) is int:
                id_str = ','.join(str(x) for x in list(range(data_parallel_device_ids)))
                os.environ["CUDA_VISIBLE_DEVICES"] = id_str
            else:
                gpu_str = ','.join([str(x) for x in data_parallel_device_ids])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        print(f'VISIBLE GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    def ddp_train(self, gpu_nb, model):
        """
        Entry point into a DP thread
        :param gpu_nb:
        :param model:
        :param cluster_obj:
        :return:
        """
        # node rank using relative slurm id
        # otherwise default to node rank 0
        try:
            node_id = os.environ['SLURM_NODEID']
            self.node_rank = int(node_id)
        except Exception:
            self.node_rank = 0

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_nb == 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_nb
            self.world_size = self.nb_gpu_nodes * self.num_gpus

        elif self.use_ddp2:
            self.proc_rank = self.node_rank
            self.world_size = self.nb_gpu_nodes

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.__init_tcp_connection()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # set model properties before going into wrapper
        self.copy_trainer_model_properties(model)

        # override root GPU
        self.root_gpu = gpu_nb

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        # DDP2 uses all GPUs on the machine
        if self.distributed_backend == 'ddp':
            device_ids = [gpu_nb]
        elif self.use_ddp2:
            device_ids = None

        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )

        # continue training routine
        self.run_pretrain_routine(model)

    def __init_tcp_connection(self):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        :param port:
        :param tries:
        :return:
        """

        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ['SLURM_JOB_ID']
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception as e:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.2'

        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group("nccl", rank=self.proc_rank, world_size=self.world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node
