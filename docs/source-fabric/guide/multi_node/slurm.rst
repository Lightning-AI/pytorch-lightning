:orphan:

##############################
Run on a SLURM Managed Cluster
##############################

**Audience**: Users who need to run on an academic or enterprise private cluster.

Lightning automates the details behind training on a SLURM-powered cluster.
Unlike the :doc:`general-purpose cluster <./barebones>`, with SLURM the users don't need to start the jobs manually on each node but instead submit it to SLURM, which schedules the resources and time for which the job is allowed to run.

Don't have access to an enterprise cluster? Try the :doc:`Lightning cloud <./cloud>`.

----


*********************************
Submit a training script to SLURM
*********************************

To train a model using multiple nodes, do the following:

**Step 1:** Set the number of devices per node and how many nodes the training will run on.

.. code-block:: python

    from lightning.fabric import Fabric

    # Train on 32 GPUs across 4 nodes
    fabric = Fabric(accelerator="gpu", devices=8, num_nodes=4)

By default, this will run classic *distributed data-parallel*.
Optionally, explore other strategies too:

.. code-block:: python

    # DeepSpeed
    fabric = Fabric(accelerator="gpu", devices=8, num_nodes=4, strategy="deepspeed")

    # Fully Sharded Data Parallel (FSDP)
    fabric = Fabric(accelerator="gpu", devices=8, num_nodes=4, strategy="fsdp")


**Step 2:** Call :meth:`~lightning.fabric.fabric.Fabric.launch` to initialize the communication between devices and nodes.

.. code-block:: python

    fabric = Fabric(...)
    fabric.launch()


**Step 3:** Create the appropriate SLURM job configuration:

.. code-block:: bash
    :caption: submit.sh
    :emphasize-lines: 4,5,21

    #!/bin/bash -l

    # SLURM SUBMIT SCRIPT
    #SBATCH --nodes=4               # This needs to match Fabric(num_nodes=...)
    #SBATCH --ntasks-per-node=8     # This needs to match Fabric(devices=...)
    #SBATCH --gres=gpu:8            # Request N GPUs per machine
    #SBATCH --mem=0
    #SBATCH --time=0-02:00:00

    # Activate conda environment
    source activate $1

    # Debugging flags (optional)
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1

    # On your cluster you might need this:
    # export NCCL_SOCKET_IFNAME=^docker0,lo

    # Run your training script
    srun python train.py


**Step 4:** Submit the job to SLURM

.. code-block:: bash

    sbatch submit.sh


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

**My program is stuck initializing at startup. What is causing this?**

You are seeing a message like this in the logs, but nothing happens:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4


The most likely reasons and how to fix it:

- You forgot to run the ``python train.py`` command with ``srun``:
  Please have a look at the SLURM template script above, which includes the ``srun`` at the bottom of the script.

- The number of nodes or the number of devices per node is misconfigured:
  Two parameters in the SLURM submission script determine how many processes will run your training, the ``#SBATCH --nodes=X`` setting and ``#SBATCH --ntasks-per-node=Y`` settings.
  The numbers there need to match what is configured in Fabric in the code: ``Fabric(num_nodes=X, devices=Y)``.
  If you change the numbers, update them in BOTH places.


If you are sick of troubleshooting SLURM settings, give :doc:`Lightning cloud <./cloud>` a try!
For other questions, please don't hesitate to join the `Discord <https://discord.gg/VptPCZkGNa>`_.
