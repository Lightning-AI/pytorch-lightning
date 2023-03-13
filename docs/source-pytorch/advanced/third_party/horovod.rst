:orphan:

#######
Horovod
#######

The `Horovod strategy <https://github.com/Lightning-AI/lightning-horovod>`_ allows the same training script to be used for single-GPU, multi-GPU, and multi-node training.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Like Distributed Data Parallel, every process in Horovod operates on a single GPU with a fixed subset of the data.  Gradients are averaged across all GPUs in parallel during the backward pass, then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`). In the training script, Horovod will detect the number of workers from the environment, and automatically scale the learning rate to compensate for the increased total batch size.

You can install the Horovod integration by running

.. code-block:: bash

    pip install lightning-horovod

This will install both the `Horovod <https://github.com/horovod/horovod#install>`_ package as well as the ``HorovodStrategy`` for the Lightning Trainer.
Horovod can be configured in the training script to run with any number of GPUs / processes as follows:

.. code-block:: python

    # train Horovod on CPU (number of processes / machines provided on command-line)
    trainer = Trainer(strategy=HorovodStrategy())

When starting the training job, the driver application will then be used to specify the total number of worker processes:


.. code-block:: bash

    # run training with 4 GPUs on a single machine
    horovodrun -np 4 python train.py

    # run training with 8 GPUs on two machines (4 GPUs each)
    horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py


See the official [Horovod documentation](https://horovod.readthedocs.io/en/stable) for details on installation and performance tuning.
