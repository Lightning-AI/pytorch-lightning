import os
import re
import signal
import pdb
from subprocess import call

import torch
import torch.distributed as dist
from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)


class TrainerDDPMixin(object):

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
        self.__copy_trainer_model_properties(model)

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
        self.__run_pretrain_routine(model)

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
