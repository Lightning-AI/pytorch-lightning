:orphan:

.. _gpu_intermediate:

GPU training (Intermediate)
===========================
**Audience:** Users looking to train across machines or experiment with different scaling techniques.

----

Distributed Training strategies
-------------------------------
Lightning supports multiple ways of doing distributed training.

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+4-+multi+node+training_3.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_multi_gpus.png
    :width: 400

- DistributedDataParallel (multiple-gpus across many machines)
    - Regular (``strategy='ddp'``)
    - Spawn (``strategy='ddp_spawn'``)
    - Notebook/Fork (``strategy='ddp_notebook'``)

.. note::
    If you request multiple GPUs or nodes without setting a strategy, DDP will be automatically used.

For a deeper understanding of what Lightning is doing, feel free to read this
`guide <https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565>`_.


Distributed Data Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~torch.nn.parallel.DistributedDataParallel` (DDP) works as follows:

1. Each GPU across each node gets its own process.

2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.

3. Each process inits the model.

4. Each process performs a full forward and backward pass in parallel.

5. The gradients are synced and averaged across all processes.

6. Each process updates its optimizer.

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp")

    # train on 32 GPUs (4 nodes)
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp", num_nodes=4)

This Lightning implementation of DDP calls your script under the hood multiple times with the correct environment
variables:

.. code-block:: bash

    # example for 3 GPUs DDP
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc

We use DDP this way because `ddp_spawn` has a few limitations (due to Python and PyTorch):

1. Since `.spawn()` trains the model in subprocesses, the model on the main process does not get updated.
2. Dataloader(num_workers=N), where N is large, bottlenecks training with DDP... ie: it will be VERY slow or won't work at all. This is a PyTorch limitation.
3. Forces everything to be picklable.

There are cases in which it is NOT possible to use DDP. Examples are:

- Jupyter Notebook, Google COLAB, Kaggle, etc.
- You have a nested script without a root package

In these situations you should use `ddp_notebook` or `dp` instead.

Distributed Data Parallel Spawn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`ddp_spawn` is exactly like `ddp` except that it uses .spawn to start the training processes.

.. warning:: It is STRONGLY recommended to use `DDP` for speed and performance.

.. code-block:: python

    mp.spawn(self.ddp_train, nprocs=self.num_processes, args=(model,))

If your script does not support being called from the command line (ie: it is nested without a root
project module) you can use the following method:

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp_spawn")

We STRONGLY discourage this use because it has limitations (due to Python and PyTorch):

1. The model you pass in will not update. Please save a checkpoint and restore from there.
2. Set Dataloader(num_workers=0) or it will bottleneck training.

`ddp` is MUCH faster than `ddp_spawn`. We recommend you

1. Install a top-level module for your project using setup.py

.. code-block:: python

    # setup.py
    #!/usr/bin/env python

    from setuptools import setup, find_packages

    setup(
        name="src",
        version="0.0.1",
        description="Describe Your Cool Project",
        author="",
        author_email="",
        url="https://github.com/YourSeed",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
        install_requires=["lightning"],
        packages=find_packages(),
    )

2. Setup your project like so:

.. code-block:: bash

    /project
        /src
            some_file.py
            /or_a_folder
        setup.py

3. Install as a root-level package

.. code-block:: bash

    cd /project
    pip install -e .

You can then call your scripts anywhere

.. code-block:: bash

    cd /project/src
    python some_file.py --accelerator 'gpu' --devices 8 --strategy 'ddp'


Distributed Data Parallel in Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DDP Notebook/Fork is an alternative to Spawn that can be used in interactive Python and Jupyter notebooks, Google Colab, Kaggle notebooks, and so on:
The Trainer enables it by default when such environments are detected.

.. code-block:: python

    # train on 8 GPUs in a Jupyter notebook
    trainer = Trainer(accelerator="gpu", devices=8)

    # can be set explicitly
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp_notebook")

    # can also be used in non-interactive environments
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp_fork")

Among the native distributed strategies, regular DDP (``strategy="ddp"``) is still recommended as the go-to strategy over Spawn and Fork/Notebook for its speed and stability but it can only be used with scripts.


Comparison of DDP variants and tradeoffs
****************************************

.. list-table:: DDP variants and their tradeoffs
   :widths: 40 20 20 20
   :header-rows: 1

   * -
     - DDP
     - DDP Spawn
     - DDP Notebook/Fork
   * - Works in Jupyter notebooks / IPython environments
     - No
     - No
     - Yes
   * - Supports multi-node
     - Yes
     - Yes
     - Yes
   * - Supported platforms
     - Linux, Mac, Win
     - Linux, Mac, Win
     - Linux, Mac
   * - Requires all objects to be picklable
     - No
     - Yes
     - No
   * - Limitations in the main process
     - None
     - The state of objects is not up-to-date after returning to the main process (`Trainer.fit()` etc). Only the model parameters get transferred over.
     - GPU operations such as moving tensors to the GPU or calling ``torch.cuda`` functions before invoking ``Trainer.fit`` is not allowed.
   * - Process creation time
     - Slow
     - Slow
     - Fast


Distributed and 16-bit precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are the possible configurations we support.

+-------+---------+-----+--------+-----------------------------------------------------------------------+
| 1 GPU | 1+ GPUs | DDP | 16-bit | command                                                               |
+=======+=========+=====+========+=======================================================================+
| Y     |         |     |        | `Trainer(accelerator="gpu", devices=1)`                               |
+-------+---------+-----+--------+-----------------------------------------------------------------------+
| Y     |         |     | Y      | `Trainer(accelerator="gpu", devices=1, precision=16)`                 |
+-------+---------+-----+--------+-----------------------------------------------------------------------+
|       | Y       | Y   |        | `Trainer(accelerator="gpu", devices=k, strategy='ddp')`               |
+-------+---------+-----+--------+-----------------------------------------------------------------------+
|       | Y       | Y   | Y      | `Trainer(accelerator="gpu", devices=k, strategy='ddp', precision=16)` |
+-------+---------+-----+--------+-----------------------------------------------------------------------+

DDP can also be used with 1 GPU, but there's no reason to do so other than debugging distributed-related issues.


Implement Your Own Distributed (DDP) training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you need your own way to init PyTorch DDP you can override :meth:`lightning.pytorch.strategies.ddp.DDPStrategy.setup_distributed`.

If you also need to use your own DDP implementation, override :meth:`lightning.pytorch.strategies.ddp.DDPStrategy.configure_ddp`.

----------

Torch Distributed Elastic
-------------------------
Lightning supports the use of Torch Distributed Elastic to enable fault-tolerant and elastic distributed job scheduling. To use it, specify the 'ddp' backend and the number of GPUs you want to use in the trainer.

.. code-block:: python

    Trainer(accelerator="gpu", devices=8, strategy="ddp")

To launch a fault-tolerant job, run the following on all nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=NUM_NODES
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=c10d
            --rdzv_endpoint=HOST_NODE_ADDR
            YOUR_LIGHTNING_TRAINING_SCRIPT.py (--arg1 ... train script args...)

To launch an elastic job, run the following on at least ``MIN_SIZE`` nodes and at most ``MAX_SIZE`` nodes.

.. code-block:: bash

    python -m torch.distributed.run
            --nnodes=MIN_SIZE:MAX_SIZE
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=c10d
            --rdzv_endpoint=HOST_NODE_ADDR
            YOUR_LIGHTNING_TRAINING_SCRIPT.py (--arg1 ... train script args...)

See the official `Torch Distributed Elastic documentation <https://pytorch.org/docs/stable/distributed.elastic.html>`_ for details
on installation and more use cases.

Optimize multi-machine communication
------------------------------------

By default, Lightning will select the ``nccl`` backend over ``gloo`` when running on GPUs.
Find more information about PyTorch's supported backends `here <https://pytorch.org/docs/stable/distributed.html>`__.

Lightning allows explicitly specifying the backend via the `process_group_backend` constructor argument on the relevant Strategy classes. By default, Lightning will select the appropriate process group backend based on the hardware used.

.. code-block:: python

    from lightning.pytorch.strategies import DDPStrategy

    # Explicitly specify the process group backend if you choose to
    ddp = DDPStrategy(process_group_backend="nccl")

    # Configure the strategy on the Trainer
    trainer = Trainer(strategy=ddp, accelerator="gpu", devices=8)
