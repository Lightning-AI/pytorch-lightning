.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _slurm:

Computing cluster (SLURM)
=========================

Lightning automates the details behind training on a SLURM-powered cluster.

.. _multi-node:

----------

Multi-node training
-------------------
To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module`.

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(gpus=8, num_nodes=4, accelerator='ddp')

3.  It's a good idea to structure your training script like this:

    .. testcode::

        # train.py
        def main(hparams):
            model = LightningTemplateModel(hparams)

            trainer = pl.Trainer(
                gpus=8,
                num_nodes=4,
                accelerator='ddp'
            )

            trainer.fit(model)


        if __name__ == '__main__':
            root_dir = os.path.dirname(os.path.realpath(__file__))
            parent_parser = ArgumentParser(add_help=False)
            hyperparams = parser.parse_args()

            # TRAIN
            main(hyperparams)

4.  Create the appropriate SLURM job:

    .. code-block:: bash

        # (submit.sh)
        #!/bin/bash -l

        # SLURM SUBMIT SCRIPT
        #SBATCH --nodes=4
        #SBATCH --gres=gpu:8
        #SBATCH --ntasks-per-node=8
        #SBATCH --mem=0
        #SBATCH --time=0-02:00:00

        # activate conda env
        source activate $1

        # debugging flags (optional)
        export NCCL_DEBUG=INFO
        export PYTHONFAULTHANDLER=1

        # on your cluster you might need these:
        # set the network interface
        # export NCCL_SOCKET_IFNAME=^docker0,lo

        # might need the latest CUDA
        # module load NCCL/2.4.7-1-cuda.10.0

        # run script from above
        srun python3 train.py

5.  If you want auto-resubmit (read below), add this line to the submit.sh script

    .. code-block:: bash

        #SBATCH --signal=SIGUSR1@90

6.  Submit the SLURM job

    .. code-block:: bash

        sbatch submit.sh

.. note::
    When running in DDP mode, any errors in your code will show up as an NCCL issue.
    Set the `NCCL_DEBUG=INFO` flag to see the ACTUAL error.


Normally now you would need to add a
:class:`~torch.utils.data.distributed.DistributedSampler` to your dataset, however
Lightning automates this for you. But if you still need to set a sampler set the Trainer flag
:paramref:`~pytorch_lightning.Trainer.replace_sampler_ddp` to ``False``.

Here's an example of how to add your own sampler (again, not needed with Lightning).

.. testcode::

    # in your LightningModule
    def train_dataloader(self):
        dataset = MyDataset()
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = Dataloader(dataset, sampler=dist_sampler)
        return dataloader

    # in your training script
    trainer = Trainer(replace_sampler_ddp=False)

----------

Wall time auto-resubmit
-----------------------
When you use Lightning in a SLURM cluster, it automatically detects when it is about
to run into the wall time and does the following:

1.  Saves a temporary checkpoint.
2.  Requeues the job.
3.  When the job starts, it loads the temporary checkpoint.

To get this behavior make sure to add the correct signal to your SLURM script

.. code-block:: bash

    # 90 seconds before training ends
    SBATCH --signal=SIGUSR1@90

----------

Building SLURM scripts
----------------------

Instead of manually building SLURM scripts, you can use the
`SlurmCluster object <https://williamfalcon.github.io/test-tube/hpc/SlurmCluster>`_
to do this for you. The SlurmCluster can also run a grid search if you pass
in a `HyperOptArgumentParser
<https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser>`_.

Here is an example where you run a grid search of 9 combinations of hyperparameters.
See also the multi-node examples
`here <https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pl_examples/basic_examples>`__.

.. code-block:: python

    # grid search 3 values of learning rate and 3 values of number of layers for your net
    # this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32),
    # (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.opt_list('--learning_rate', default=0.001, type=float,
                    options=[1e-3, 1e-2, 1e-1], tunable=True)
    parser.opt_list('--layers', default=1, type=float, options=[16, 32, 64], tunable=True)
    hyperparams = parser.parse_args()

    # Slurm cluster submits 9 jobs, each with a set of hyperparams
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/some/path/to/save',
    )

    # OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
    # which interface your nodes use for communication
    cluster.add_command('export NCCL_SOCKET_IFNAME=^docker0,lo')

    # see the output of the NCCL connection process
    # NCCL is how the nodes talk to each other
    cluster.add_command('export NCCL_DEBUG=INFO')

    # setting a master port here is a good idea.
    cluster.add_command('export MASTER_PORT=%r' % PORT)

    # ************** DON'T FORGET THIS ***************
    # MUST load the latest NCCL version
    cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

    # configure cluster
    cluster.per_experiment_nb_nodes = 12
    cluster.per_experiment_nb_gpus = 8

    cluster.add_slurm_cmd(cmd='ntasks-per-node', value=8, comment='1 task per gpu')

    # submit a script with 9 combinations of hyper params
    # (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=9, # how many permutations of the grid search to run
        job_name='name_for_squeue'
    )


The other option is that you generate scripts on your own via a bash command or use another library.

----------

Self-balancing architecture (COMING SOON)
-----------------------------------------

Here Lightning distributes parts of your module across available GPUs to optimize for speed and memory.
