####################################
Run on an on-prem cluster (advanced)
####################################

.. _slurm:

----

******************************
Run on a SLURM-managed cluster
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
        def main(args):
            model = YourLightningModule(args)

            trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")

            trainer.fit(model)


        if __name__ == "__main__":
            args = ...  # you can use your CLI parser of choice, or the `LightningCLI`
            # TRAIN
            main(args)

4.  Create the appropriate SLURM job:

    .. code-block:: bash

        # (submit.sh)
        #!/bin/bash -l

        # SLURM SUBMIT SCRIPT
        #SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
        #SBATCH --gres=gpu:8
        #SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
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

5.  If you want to auto-resubmit (read below), add this line to the submit.sh script

    .. code-block:: bash

        #SBATCH --signal=SIGUSR1@90

6.  Submit the SLURM job

    .. code-block:: bash

        sbatch submit.sh

----

***********************************
Enable auto wall-time resubmissions
***********************************
When you use Lightning in a SLURM cluster, it automatically detects when it is about
to run into the wall time and does the following:

1.  Saves a temporary checkpoint.
2.  Requeues the job.
3.  When the job starts, it loads the temporary checkpoint.

To get this behavior make sure to add the correct signal to your SLURM script

.. code-block:: bash

    # 90 seconds before training ends
    SBATCH --signal=SIGUSR1@90

You can change this signal if your environment requires the use of a different one, for example

.. code-block:: bash

    #SBATCH --signal=SIGHUP@90

Then, when you make your trainer, pass the `requeue_signal` option to the :class:`~lightning.pytorch.plugins.environments.slurm_environment.SLURMEnvironment` plugin:

.. code-block:: python

    trainer = Trainer(plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)])

If auto-resubmit is not desired, it can be turned off in the :class:`~lightning.pytorch.plugins.environments.slurm_environment.SLURMEnvironment` plugin:

.. code-block:: python

    from lightning.pytorch.plugins.environments import SLURMEnvironment

    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=False)])

----


****************
Interactive Mode
****************

You can also let SLURM schedule a machine for you and then log in to the machine to run scripts manually.
This is useful for development and debugging.
If you set the job name to *bash* or *interactive*, and then log in and run scripts, Lightning's SLURM auto-detection will get bypassed and it can launch processes normally:

.. code-block:: bash

    # make sure to set `--job-name "interactive"`
    srun --account <your-account> --pty bash --job-name "interactive" ...

    # now run scripts normally
    python train.py ...


----


***************
Troubleshooting
***************

**The Trainer is stuck initializing at startup, what is causing this?**

You are seeing a message like this in the logs but nothing happens:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4


The most likely reasons and how to fix it:

- You forgot to run the ``python train.py`` command with ``srun``:
  Please have a look at the SLURM template script above which includes the ``srun`` at the bottom of the script.

- The number of nodes or number of devices per node is configured incorrectly:
  There are two parameters in the SLURM submission script that determine how many processes will run your training, the ``#SBATCH --nodes=X`` setting and ``#SBATCH --ntasks-per-node=Y`` settings.
  The numbers there need to match what is configured in your Trainer in the code: ``Trainer(num_nodes=X, devices=Y)``.
  If you change the numbers, update them in BOTH places.
