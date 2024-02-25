:orphan:

.. _gpu_intermediate:

GPU training (Intermediate)
===========================
**Audience:** Users looking to train across machines or experiment with different scaling techniques.

----


Distributed training strategies
-------------------------------
Lightning supports multiple ways of doing distributed training.

- Regular (``strategy='ddp'``)
- Spawn (``strategy='ddp_spawn'``)
- Notebook/Fork (``strategy='ddp_notebook'``)

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+4-+multi+node+training_3.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_multi_gpus.png
    :width: 400


.. note::
    If you request multiple GPUs or nodes without setting a strategy, DDP will be automatically used.

For a deeper understanding of what Lightning is doing, feel free to read this
`guide <https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565>`_.


----


Distributed Data Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`~torch.nn.parallel.DistributedDataParallel` (DDP) works as follows:

1. Each GPU across each node gets its own process.
2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.
3. Each process inits the model.
4. Each process performs a full forward and backward pass in parallel.
5. The gradients are synced and averaged across all processes.
6. Each process updates its optimizer.

|

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
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=1 python my_file.py --accelerator 'gpu' --devices 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=2 python my_file.py --accelerator 'gpu' --devices 3 --etc

Using DDP this way has a few disadvantages over ``torch.multiprocessing.spawn()``:

1. All processes (including the main process) participate in training and have the updated state of the model and Trainer state.
2. No multiprocessing pickle errors
3. Easily scales to multi-node training

|

It is NOT possible to use DDP in interactive environments like Jupyter Notebook, Google COLAB, Kaggle, etc.
In these situations you should use `ddp_notebook`.


----


Distributed Data Parallel Spawn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: It is STRONGLY recommended to use DDP for speed and performance.

The `ddp_spawn` strategy is similar to `ddp` except that it uses ``torch.multiprocessing.spawn()`` to start the training processes.
Use this for debugging only, or if you are converting a code base to Lightning that relies on spawn.

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp_spawn")

We STRONGLY discourage this use because it has limitations (due to Python and PyTorch):

1. After ``.fit()``, only the model's weights get restored to the main process, but no other state of the Trainer.
2. Does not support multi-node training.
3. It is generally slower than DDP.


----


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


----


Comparison of DDP variants and tradeoffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


----


TorchRun (TorchElastic)
-----------------------
Lightning supports the use of TorchRun (previously known as TorchElastic) to enable fault-tolerant and elastic distributed job scheduling.
To use it, specify the DDP strategy and the number of GPUs you want to use in the Trainer.

.. code-block:: python

    Trainer(accelerator="gpu", devices=8, strategy="ddp")

Then simply launch your script with the :doc:`torchrun <../clouds/cluster_intermediate_2>` command.


----


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
