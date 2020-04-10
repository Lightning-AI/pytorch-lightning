"""
Lightning supports model training on a cluster managed by SLURM in the following cases:

1. Training on a single cpu or single GPU.
2. Train on multiple GPUs on the same node using DataParallel or DistributedDataParallel
3. Training across multiple GPUs on multiple different nodes via DistributedDataParallel.

.. note:: A node means a machine with multiple GPUs

Running grid search on a cluster
--------------------------------

To use lightning to run a hyperparameter search (grid-search or random-search) on a cluster do 4 things:

(1). Define the parameters for the grid search

.. code-block:: python

    from test_tube import HyperOptArgumentParser

    # subclass of argparse
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')

    # let's enable optimizing over the number of layers in the network
    parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])

    hparams = parser.parse_args()

.. note:: You must set `Tunable=True` for that argument to be considered in the permutation set.
 Otherwise test-tube will use the default value. This flag is useful when you don't want
 to search over an argument and want to use the default instead.

(2). Define the cluster options in the
 `SlurmCluster object <https://williamfalcon.github.io/test-tube/hpc/SlurmCluster>`_ (over 5 nodes and 8 gpus)

.. code-block:: python

    from test_tube.hpc import SlurmCluster

    # hyperparameters is a test-tube hyper params object
    # see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
    hyperparams = args.parse()

    # init cluster
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/path/to/log/results/to',
        python_cmd='python3'
    )

    # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
    cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)

    # set the job options. In this instance, we'll run 20 different models
    # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
    cluster.per_experiment_nb_gpus = 8
    cluster.per_experiment_nb_nodes = 5

    # we'll request 10GB of memory per node
    cluster.memory_mb_per_node = 10000

    # set a walltime of 10 minues
    cluster.job_time = '10:00'


(3). Make a main function with your model and trainer. Each job will call this function with a particular
hparams configuration.::

    from pytorch_lightning import Trainer

    def train_fx(trial_hparams, cluster_manager, _):
        # hparams has a specific set of hyperparams

        my_model = MyLightningModel()

        # give the trainer the cluster object
        trainer = Trainer()
        trainer.fit(my_model)

    `

(4). Start the grid/random search::

    # run the models on the cluster
    cluster.optimize_parallel_cluster_gpu(
        train_fx,
        nb_trials=20,
        job_name='my_grid_search_exp_name',
        job_display_name='my_exp')

.. note:: `nb_trials` specifies how many of the possible permutations to use. If using `grid_search` it will use
 the depth first ordering. If using `random_search` it will use the first k shuffled options. FYI, random search
 has been shown to be just as good as any Bayesian optimization method when using a reasonable number of samples (60),
 see this `paper <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_  for more information.

Walltime auto-resubmit
----------------------

Lightning automatically resubmits jobs when they reach the walltime. Make sure to set the SIGUSR1 signal in
your SLURM script.::

    # 90 seconds before training ends
    #SBATCH --signal=SIGUSR1@90

When lightning receives the SIGUSR1 signal it will:
1. save a checkpoint with 'hpc_ckpt' in the name.
2. resubmit the job using the SLURM_JOB_ID

When the script starts again, Lightning will:
1. search for a 'hpc_ckpt' checkpoint.
2. restore the model, optimizers, schedulers, epoch, etc...

"""

import os
import re
from abc import ABC, abstractmethod
from typing import Union

import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import set_proc_rank, rank_zero_warn

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True


class TrainerDDPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    num_gpu_nodes: int
    logger: Union[LightningLoggerBase, bool]
    data_parallel_device_ids: ...
    distributed_backend: str
    amp_level: str
    use_tpu: bool
    default_root_dir: str

    @property
    @abstractmethod
    def num_gpus(self) -> int:
        """Warning: this is just empty shell for code implemented in other class."""

    @property
    @abstractmethod
    def use_amp(self) -> bool:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def copy_trainer_model_properties(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def run_pretrain_routine(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def init_optimizers(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def init_tpu(self):
        # turn off all the GPU stuff
        self.distributed_backend = None

        # enable tpu
        self.use_tpu = True

    def set_distributed_mode(self, distributed_backend, num_gpu_nodes):
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
                rank_zero_warn('You requested multiple GPUs but did not specify a backend, e.g.'
                               ' Trainer(distributed_backend=dp) (or ddp, ddp2).'
                               ' Setting distributed_backend=dp for you.')
                self.use_dp = True
                self.use_ddp = False
                self.use_ddp2 = False

        # throw error to force user ddp or ddp2 choice
        if num_gpu_nodes > 1 and not (self.use_ddp2 or self.use_ddp):
            raise MisconfigurationException(
                'DataParallel does not support num_nodes > 1. Switching to DistributedDataParallel for you. '
                'To silence this warning set distributed_backend=ddp or distributed_backend=ddp2'
            )

        log.info(f'GPU available: {torch.cuda.is_available()}, used: {self.on_gpu}')

    def configure_slurm_ddp(self, num_gpu_nodes):
        self.is_slurm_managing_tasks = False

        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.use_ddp:
            self.num_requested_gpus = self.num_gpus * num_gpu_nodes
            self.num_slurm_tasks = 0
            try:
                self.num_slurm_tasks = int(os.environ['SLURM_NTASKS'])
                self.is_slurm_managing_tasks = self.num_slurm_tasks == self.num_requested_gpus

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
            if isinstance(data_parallel_device_ids, int):
                id_str = ','.join(str(x) for x in list(range(data_parallel_device_ids)))
                os.environ["CUDA_VISIBLE_DEVICES"] = id_str
            else:
                gpu_str = ','.join([str(x) for x in data_parallel_device_ids])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        log.info(f'VISIBLE GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    def ddp_train(self, gpu_idx, model):
        """
        Entry point into a DP thread
        :param gpu_idx:
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
        self.progress_bar_refresh_rate = self.progress_bar_refresh_rate if self.node_rank == 0 and gpu_idx == 0 else 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_idx
            self.world_size = self.num_gpu_nodes * self.num_gpus

        elif self.use_ddp2:
            self.proc_rank = self.node_rank
            self.world_size = self.num_gpu_nodes
        # set warning rank
        set_proc_rank(self.proc_rank)

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self
        model.init_ddp_connection(self.proc_rank, self.world_size)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_idx)
        model.cuda(gpu_idx)

        # set model properties before going into wrapper
        self.copy_trainer_model_properties(model)

        # override root GPU
        self.root_gpu = gpu_idx

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
            self.optimizers = optimizers

        # DDP2 uses all GPUs on the machine
        if self.distributed_backend == 'ddp':
            device_ids = [gpu_idx]
        elif self.use_ddp2:
            device_ids = self.data_parallel_device_ids
        else:
            device_ids = None

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        self.run_pretrain_routine(model)

        # when ddp ends, we save the model
        self.save_spawn_weights(model)

    def save_spawn_weights(self, model):
        """
        Dump a temporary checkpoint after ddp ends to get weights out of the process
        :param model:
        :return:
        """
        if self.proc_rank == 0:
            path = os.path.join(self.default_root_dir, '__temp_weight_ddp_end.ckpt')
            self.save_checkpoint(path)

    def load_spawn_weights(self, original_model):
        """
        Load the temp weights saved in the process
        To recover the trained model from the ddp process we load the saved weights
        :param model:
        :return:
        """

        loaded_model = original_model

        if self.proc_rank == 0:
            # load weights saved in ddp
            path = os.path.join(self.default_root_dir, '__temp_weight_ddp_end.ckpt')
            loaded_model = original_model.__class__.load_from_checkpoint(path)

            # copy loaded weights to old model
            original_model.load_state_dict(loaded_model.state_dict())

            # remove ddp weights
            os.remove(path)

        return loaded_model

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node
