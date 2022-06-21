####################################
Run on an on-prem cluster (advanced)
####################################

.. _slurm:

----

******************************
Run on a SLURM managed cluster
******************************
Lightning automates the details behind training on a SLURM-powered cluster. In contrast to the general purpose
cluster above, the user does not start the jobs manually on each node and instead submits it to SLURM which
schedules the resources and time for which the job is allowed to run.

----

***************************
Design your training script
***************************

To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module` (no need to add anything specific here).

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")

3.  It's a good idea to structure your training script like this:

    .. testcode::

        # train.py
        def main(hparams):
            model = LightningTemplateModel(hparams)

            trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")

            trainer.fit(model)


        if __name__ == "__main__":
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

----

**********************************
Enable auto wall-time resubmitions
**********************************
When you use Lightning in a SLURM cluster, it automatically detects when it is about
to run into the wall time and does the following:

1.  Saves a temporary checkpoint.
2.  Requeues the job.
3.  When the job starts, it loads the temporary checkpoint.

To get this behavior make sure to add the correct signal to your SLURM script

.. code-block:: bash

    # 90 seconds before training ends
    SBATCH --signal=SIGUSR1@90

If auto-resubmit is not desired, it can be turned off in the :class:`~pytorch_lightning.plugins.environments.slurm_environment.SLURMEnvironment` plugin:

.. code-block:: python

    from pytorch_lightning.plugins.environments import SLURMEnvironment

    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=False)])

----

***********************
Build your SLURM script
***********************
Instead of manually building SLURM scripts, you can use the
`SlurmCluster object <https://williamfalcon.github.io/test-tube/hpc/SlurmCluster>`_
to do this for you. The SlurmCluster can also run a grid search if you pass
in a `HyperOptArgumentParser
<https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser>`_.

Here is an example where you run a grid search of 9 combinations of hyperparameters.
See also the multi-node examples
`here <https://github.com/PyTorchLightning/pytorch-lightning/tree/master/examples/pl_basics>`__.

.. code-block:: python

    # grid search 3 values of learning rate and 3 values of number of layers for your net
    # this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32),
    # (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
    parser.opt_list("--learning_rate", default=0.001, type=float, options=[1e-3, 1e-2, 1e-1], tunable=True)
    parser.opt_list("--layers", default=1, type=float, options=[16, 32, 64], tunable=True)
    hyperparams = parser.parse_args()

    # Slurm cluster submits 9 jobs, each with a set of hyperparams
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path="/some/path/to/save",
    )

    # OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
    # which interface your nodes use for communication
    cluster.add_command("export NCCL_SOCKET_IFNAME=^docker0,lo")

    # see the output of the NCCL connection process
    # NCCL is how the nodes talk to each other
    cluster.add_command("export NCCL_DEBUG=INFO")

    # setting a main port here is a good idea.
    cluster.add_command("export MASTER_PORT=%r" % PORT)

    # ************** DON'T FORGET THIS ***************
    # MUST load the latest NCCL version
    cluster.load_modules(["NCCL/2.4.7-1-cuda.10.0"])

    # configure cluster
    cluster.per_experiment_nb_nodes = 12
    cluster.per_experiment_nb_gpus = 8

    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=8, comment="1 task per gpu")

    # submit a script with 9 combinations of hyper params
    # (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=9, job_name="name_for_squeue"  # how many permutations of the grid search to run
    )


The other option is that you generate scripts on your own via a bash command or use our
:doc:`native solution <../clouds/cloud_training>`.

----

********
Get help
********
Setting up a cluster for distributed training is not trivial. Lightning offers lightning-grid which allows you to configure a cluster easily and run experiments via the CLI and web UI.

Try it out for free today:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Train models on the cloud
   :description: Learn to run a model in the background on a cloud machine.
   :col_css: col-md-6
   :button_link: cloud_training.html
   :height: 150
   :tag: intermediate

.. raw:: html

        </div>
    </div
