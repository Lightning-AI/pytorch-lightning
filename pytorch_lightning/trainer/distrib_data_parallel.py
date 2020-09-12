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
from typing import Union, List, Optional, Tuple

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.distributed import rank_zero_warn, rank_zero_info

try:
    from apex import amp
except ImportError:
    amp = None


class TrainerDDPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    num_gpu_nodes: int
    gpus: List[int]
    logger: Union[LightningLoggerBase, bool]
    data_parallel_device_ids: ...
    distributed_backend: Optional[str]
    amp_level: str
    use_tpu: bool
    default_root_dir: str
    progress_bar_callback: ...
    checkpoint_callback: ...
    num_processes: int
    num_nodes: int
    node_rank: int
    tpu_cores: int
    testing: bool
    global_rank: int
    datamodule: Optional[LightningDataModule]

    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def call_setup_hook(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @property
    @abstractmethod
    def num_gpus(self) -> int:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def init_optimizers(self, *args) -> Tuple[List, List, List]:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reinit_scheduler_properties(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def save_checkpoint(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def setup(self, *args) -> None:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_function_implemented(self, *args) -> bool:
        """Warning: this is just empty shell for code implemented in other class."""

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):
        if self.distributed_backend.lower() not in ['ddp_spawn', 'ddp_cpu', 'tpu']:
            return

        # track the best model path
        best_model_path = None
        if self.checkpoint_callback is not None:
            best_model_path = self.checkpoint_callback.best_model_path

        if self.global_rank == 0 and mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')
            # todo, pass complete checkpoint as state dictionary
            mp_queue.put(best_model_path)
            mp_queue.put(results)

            # save the last weights
            last_path = None
            if not self.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub('.ckpt', '.tmp_end.ckpt', best_model_path)
                atomic_save(model.state_dict(), last_path)
            mp_queue.put(last_path)

    def save_spawn_weights(self, model):
        """
        Dump a temporary checkpoint after ddp ends to get weights out of the process
        :param model:
        :return:
        """
        if self.is_global_zero:
            path = os.path.join(self.default_root_dir, '__temp_weight_distributed_end.ckpt')
            self.save_checkpoint(path)
            return path

    def load_spawn_weights(self, original_model):
        """
        Load the temp weights saved in the process
        To recover the trained model from the ddp process we load the saved weights
        :param model:
        :return:
        """

        loaded_model = original_model

        if self.is_global_zero:
            # load weights saved in ddp
            path = os.path.join(self.default_root_dir, '__temp_weight_distributed_end.ckpt')
            loaded_model = original_model.__class__.load_from_checkpoint(path)

            # copy loaded weights to old model
            original_model.load_state_dict(loaded_model.state_dict())

            # remove ddp weights
            os.remove(path)

        return loaded_model
