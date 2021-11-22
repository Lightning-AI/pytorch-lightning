.. testsetup:: *

    import torch
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _multi_gpu:

Multi-GPU training
==================
Lightning supports multiple ways of doing distributed training.

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_multi_gpus.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+4-+multi+node+training_3.mp4"></video>

|

----------

Preparing your code
-------------------
To train on CPU/GPU/TPU without changing your code, we need to build a few good habits :)

Delete .cuda() or .to() calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delete any calls to .cuda() or .to(device).

.. testcode::

    # before lightning
    def forward(self, x):
        x = x.cuda(0)
        layer_1.cuda(0)
        x_hat = layer_1(x)


    # after lightning
    def forward(self, x):
        x_hat = layer_1(x)

Init tensors using type_as and register_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you need to create a new tensor, use `type_as`.
This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.

.. testcode::

    # before lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.cuda(0)


    # with lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.type_as(x)

The :class:`~pytorch_lightning.core.lightning.LightningModule` knows what device it is on. You can access the reference via ``self.device``.
Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will
remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic,
register the tensor as a buffer in your modules's ``__init__`` method with :meth:`~torch.nn.Module.register_buffer`.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            ...
            self.register_buffer("sigma", torch.eye(3))
            # you can now access self.sigma anywhere in your module


Remove samplers
^^^^^^^^^^^^^^^

:class:`~torch.utils.data.distributed.DistributedSampler` is automatically handled by Lightning.

See :ref:`replace-sampler-ddp` for more information.


Synchronize validation and test logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running in distributed mode, we have to ensure that the validation and test step logging calls are synchronized across processes.
This is done by adding ``sync_dist=True`` to all ``self.log`` calls in the validation and test step.
This ensures that each GPU worker has the same behaviour when tracking model checkpoints, which is important for later downstream tasks such as testing the best checkpoint across all workers.
The ``sync_dist`` option can also be used in logging calls during the step methods, but be aware that this can lead to significant communication overhead and slow down your training.

Note if you use any built in metrics or custom metrics that use `TorchMetrics <https://torchmetrics.readthedocs.io/>`_, these do not need to be updated and are automatically handled for you.

.. testcode::

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

It is possible to perform some computation manually and log the reduced result on rank 0 as follows:

.. testcode::

    def test_step(self, batch, batch_idx):
        x, y = batch
        tensors = self(x)
        return tensors


    def test_epoch_end(self, outputs):
        mean = torch.mean(self.all_gather(outputs))

        # When logging only on rank 0, don't forget to add
        # ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        if self.trainer.is_global_zero:
            self.log("my_reduced_metric", mean, rank_zero_only=True)


Make models pickleable
^^^^^^^^^^^^^^^^^^^^^^
It's very likely your code is already `pickleable <https://docs.python.org/3/library/pickle.html>`_,
in that case no change in necessary.
However, if you run a distributed model and get the following error:

.. code-block::

    self._launch(process_obj)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47,
    in _launch reduction.dump(process_obj, fp)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
    _pickle.PicklingError: Can't pickle <function <lambda> at 0x2b599e088ae8>:
    attribute lookup <lambda> on __main__ failed

This means something in your model definition, transforms, optimizer, dataloader or callbacks cannot be pickled, and the following code will fail:

.. code-block:: python

    import pickle

    pickle.dump(some_object)

This is a limitation of using multiple processes for distributed training within PyTorch.
To fix this issue, find your piece of code that cannot be pickled. The end of the stacktrace
is usually helpful.
ie: in the stacktrace example here, there seems to be a lambda function somewhere in the code
which cannot be pickled.

.. code-block::

    self._launch(process_obj)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47,
    in _launch reduction.dump(process_obj, fp)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
    _pickle.PicklingError: Can't pickle [THIS IS THE THING TO FIND AND DELETE]:
    attribute lookup <lambda> on __main__ failed

----------

Select GPU devices
------------------

You can select the GPU devices using ranges, a list of indices or a string containing
a comma separated list of GPU ids:

.. testsetup::

    k = 1

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # DEFAULT (int) specifies how many GPUs to use per node
    Trainer(gpus=k)

    # Above is equivalent to
    Trainer(gpus=list(range(k)))

    # Specify which GPUs to use (don't use when running on cluster)
    Trainer(gpus=[0, 1])

    # Equivalent using a string
    Trainer(gpus="0, 1")

    # To use all available GPUs put -1 or '-1'
    # equivalent to list(range(torch.cuda.device_count()))
    Trainer(gpus=-1)

The table below lists examples of possible input formats and how they are interpreted by Lightning.
Note in particular the difference between `gpus=0`, `gpus=[0]` and `gpus="0"`.

+---------------+-----------+---------------------+---------------------------------+
| `gpus`        | Type      | Parsed              | Meaning                         |
+===============+===========+=====================+=================================+
| None          | NoneType  | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 0             | int       | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 3             | int       | [0, 1, 2]           | first 3 GPUs                    |
+---------------+-----------+---------------------+---------------------------------+
| -1            | int       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+
| [0]           | list      | [0]                 | GPU 0                           |
+---------------+-----------+---------------------+---------------------------------+
| [1, 3]        | list      | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "0"           | str       | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| "3"           | str       | [0, 1, 2]           | first 3 GPUs                    |
+---------------+-----------+---------------------+---------------------------------+
| "1, 3"        | str       | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "-1"          | str       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+

.. note::

    When specifying number of gpus as an integer ``gpus=k``, setting the trainer flag
    ``auto_select_gpus=True`` will automatically help you find ``k`` gpus that are not
    occupied by other processes. This is especially useful when GPUs are configured
    to be in "exclusive mode", such that only one process at a time can access them.
    For more details see the :doc:`trainer guide <../common/trainer>`.


Select torch distributed backend
--------------------------------

By default, Lightning will select the ``nccl`` backend over ``gloo`` when running on GPUs.
Find more information about PyTorch's supported backends `here <https://pytorch.org/docs/stable/distributed.html>`__.

Lightning exposes an environment variable ``PL_TORCH_DISTRIBUTED_BACKEND`` for the user to change the backend.

.. code-block:: bash

   PL_TORCH_DISTRIBUTED_BACKEND=gloo python train.py ...


----------

Distributed modes
-----------------
Lightning allows multiple ways of training

- Data Parallel (``strategy='dp'``) (multiple-gpus, 1 machine)
- DistributedDataParallel (``strategy='ddp'``) (multiple-gpus across many machines (python script based)).
- DistributedDataParallel (``strategy='ddp_spawn'``) (multiple-gpus across many machines (spawn based)).
- DistributedDataParallel 2 (``strategy='ddp2'``) (DP in a machine, DDP across machines).
- Horovod (``strategy='horovod'``) (multi-machine, multi-gpu, configured at runtime)
- TPUs (``tpu_cores=8|x``) (tpu or TPU pod)

.. note::
    If you request multiple GPUs or nodes without setting a mode, DDP Spawn will be automatically used.

For a deeper understanding of what Lightning is doing, feel free to read this
`guide <https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565>`_.



Data Parallel
^^^^^^^^^^^^^
:class:`~torch.nn.DataParallel` (DP) splits a batch across k GPUs.
That is, if you have a batch of 32 and use DP with 2 gpus, each GPU will process 16 samples,
after which the root node will aggregate the results.

.. warning:: DP use is discouraged by PyTorch and Lightning. State is not maintained on the replicas created by the
    :class:`~torch.nn.DataParallel` wrapper and you may see errors or misbehavior if you assign state to the module
    in the ``forward()`` or ``*_step()`` methods. For the same reason we cannot fully support
    :ref:`manual_optimization` with DP. Use DDP which is more stable and at least 3x faster.

.. warning:: DP only supports scattering and gathering primitive collections of tensors like lists, dicts, etc.
    Therefore the :meth:`~pytorch_lightning.core.hooks.ModelHooks.transfer_batch_to_device` hook does not apply in
    this mode and if you have overridden it, it will not be called.

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # train on 2 GPUs (using DP mode)
    trainer = Trainer(gpus=2, strategy="dp")

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
    trainer = Trainer(gpus=8, strategy="ddp")

    # train on 32 GPUs (4 nodes)
    trainer = Trainer(gpus=8, strategy="ddp", num_nodes=4)

This Lightning implementation of DDP calls your script under the hood multiple times with the correct environment
variables:

.. code-block:: bash

    # example for 3 GPUs DDP
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --gpus 3 --etc

We use DDP this way because `ddp_spawn` has a few limitations (due to Python and PyTorch):

1. Since `.spawn()` trains the model in subprocesses, the model on the main process does not get updated.
2. Dataloader(num_workers=N), where N is large, bottlenecks training with DDP... ie: it will be VERY slow or won't work at all. This is a PyTorch limitation.
3. Forces everything to be picklable.

There are cases in which it is NOT possible to use DDP. Examples are:

- Jupyter Notebook, Google COLAB, Kaggle, etc.
- You have a nested script without a root package

In these situations you should use `dp` or `ddp_spawn` instead.

Distributed Data Parallel 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In certain cases, it's advantageous to use all batches on the same machine instead of a subset.
For instance, you might want to compute a NCE loss where it pays to have more negative samples.

In  this case, we can use DDP2 which behaves like DP in a machine and DDP across nodes. DDP2 does the following:

1. Copies a subset of the data to each node.

2. Inits a model on each node.

3. Runs a forward and backward pass using DP.

4. Syncs gradients across nodes.

5. Applies the optimizer updates.

.. code-block:: python

    # train on 32 GPUs (4 nodes)
    trainer = Trainer(gpus=8, strategy="ddp2", num_nodes=4)

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
    trainer = Trainer(gpus=8, strategy="ddp_spawn")

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
        install_requires=["pytorch-lightning"],
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
    python some_file.py --accelerator 'ddp' --gpus 8


Horovod
^^^^^^^
`Horovod <http://horovod.ai>`_ allows the same training script to be used for single-GPU,
multi-GPU, and multi-node training.

Like Distributed Data Parallel, every process in Horovod operates on a single GPU with a fixed
subset of the data.  Gradients are averaged across all GPUs in parallel during the backward pass,
then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`). In
the training script, Horovod will detect the number of workers from the environment, and automatically
scale the learning rate to compensate for the increased total batch size.

Horovod can be configured in the training script to run with any number of GPUs / processes as follows:

.. code-block:: python

    # train Horovod on GPU (number of GPUs / machines provided on command-line)
    trainer = Trainer(strategy="horovod", gpus=1)

    # train Horovod on CPU (number of processes / machines provided on command-line)
    trainer = Trainer(strategy="horovod")

When starting the training job, the driver application will then be used to specify the total
number of worker processes:

.. code-block:: bash

    # run training with 4 GPUs on a single machine
    horovodrun -np 4 python train.py

    # run training with 8 GPUs on two machines (4 GPUs each)
    horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py

See the official `Horovod documentation <https://horovod.readthedocs.io/en/stable>`_ for details
on installation and performance tuning.

DP/DDP2 caveats
^^^^^^^^^^^^^^^
In DP and DDP2 each GPU within a machine sees a portion of a batch.
DP and ddp2 roughly do the following:

.. testcode::

    def distributed_forward(batch, model):
        batch = torch.Tensor(32, 8)
        gpu_0_batch = batch[:8]
        gpu_1_batch = batch[8:16]
        gpu_2_batch = batch[16:24]
        gpu_3_batch = batch[24:]

        y_0 = model_copy_gpu_0(gpu_0_batch)
        y_1 = model_copy_gpu_1(gpu_1_batch)
        y_2 = model_copy_gpu_2(gpu_2_batch)
        y_3 = model_copy_gpu_3(gpu_3_batch)

        return [y_0, y_1, y_2, y_3]

So, when Lightning calls any of the `training_step`, `validation_step`, `test_step`
you will only be operating on one of those pieces.

.. testcode::

    # the batch here is a portion of the FULL batch
    def training_step(self, batch, batch_idx):
        y_0 = batch

For most metrics, this doesn't really matter. However, if you want
to add something to your computational graph (like softmax)
using all batch parts you can use the `training_step_end` step.

.. testcode::

    def training_step_end(self, outputs):
        # only use when  on dp
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()
        return out

In pseudocode, the full sequence is:

.. code-block:: python

    # get data
    batch = next(dataloader)

    # copy model and data to each gpu
    batch_splits = split_batch(batch, num_gpus)
    models = copy_model_to_gpus(model)

    # in parallel, operate on each batch chunk
    all_results = []
    for gpu_num in gpus:
        batch_split = batch_splits[gpu_num]
        gpu_model = models[gpu_num]
        out = gpu_model(batch_split)
        all_results.append(out)

    # use the full batch for something like softmax
    full_out = model.training_step_end(all_results)

To illustrate why this is needed, let's look at DataParallel

.. testcode::

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(batch)

        # on dp or ddp2 if we did softmax now it would be wrong
        # because batch is actually a piece of the full batch
        return y_hat


    def training_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch

        # do softmax here
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()

        return out

If `training_step_end` is defined it will be called regardless of TPU, DP, DDP, etc... which means
it will behave the same regardless of the backend.

Validation and test step have the same option when using DP.

.. testcode::

    def validation_step_end(self, batch_parts_outputs):
        ...


    def test_step_end(self, batch_parts_outputs):
        ...


Distributed and 16-bit precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to an issue with Apex and DataParallel (PyTorch and NVIDIA issue), Lightning does
not allow 16-bit and DP training. We tried to get this to work, but it's an issue on their end.

Below are the possible configurations we support.

+-------+---------+----+-----+--------+------------------------------------------------------------+
| 1 GPU | 1+ GPUs | DP | DDP | 16-bit | command                                                    |
+=======+=========+====+=====+========+============================================================+
| Y     |         |    |     |        | `Trainer(gpus=1)`                                          |
+-------+---------+----+-----+--------+------------------------------------------------------------+
| Y     |         |    |     | Y      | `Trainer(gpus=1, precision=16)`                            |
+-------+---------+----+-----+--------+------------------------------------------------------------+
|       | Y       | Y  |     |        | `Trainer(gpus=k, strategy='dp')`                           |
+-------+---------+----+-----+--------+------------------------------------------------------------+
|       | Y       |    | Y   |        | `Trainer(gpus=k, strategy='ddp')`                          |
+-------+---------+----+-----+--------+------------------------------------------------------------+
|       | Y       |    | Y   | Y      | `Trainer(gpus=k, strategy='ddp', precision=16)`            |
+-------+---------+----+-----+--------+------------------------------------------------------------+


Implement Your Own Distributed (DDP) training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.plugins.training_type.ddp.DDPPlugin.init_dist_connection`.

If you also need to use your own DDP implementation, override :meth:`pytorch_lightning.plugins.training_type.ddp.DDPPlugin.configure_ddp`.


Batch size
----------
When using distributed training make sure to modify your learning rate according to your effective
batch size.

Let's say you have a batch size of 7 in your dataloader.

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):
            return Dataset(..., batch_size=7)

In DDP, DDP_SPAWN, Deepspeed, DDP_SHARDED, or Horovod your effective batch size will be 7 * gpus * num_nodes.

.. code-block:: python

    # effective batch size = 7 * 8
    Trainer(gpus=8, strategy="ddp")
    Trainer(gpus=8, strategy="ddp_spawn")
    Trainer(gpus=8, strategy="ddp_sharded")
    Trainer(gpus=8, strategy="horovod")

    # effective batch size = 7 * 8 * 10
    Trainer(gpus=8, num_nodes=10, strategy="ddp")
    Trainer(gpus=8, num_nodes=10, strategy="ddp_spawn")
    Trainer(gpus=8, num_nodes=10, strategy="ddp_sharded")
    Trainer(gpus=8, num_nodes=10, strategy="horovod")

In DDP2 or DP, your effective batch size will be 7 * num_nodes.
The reason is that the full batch is visible to all GPUs on the node when using DDP2.

.. code-block:: python

    # effective batch size = 7
    Trainer(gpus=8, strategy="ddp2")
    Trainer(gpus=8, strategy="dp")

    # effective batch size = 7 * 10
    Trainer(gpus=8, num_nodes=10, strategy="ddp2")
    Trainer(gpus=8, strategy="dp")


.. note:: Huge batch sizes are actually really bad for convergence. Check out:
        `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_

----------

Torch Distributed Elastic
-------------------------
Lightning supports the use of Torch Distributed Elastic to enable fault-tolerant and elastic distributed job scheduling. To use it, specify the 'ddp' or 'ddp2' backend and the number of gpus you want to use in the trainer.

.. code-block:: python

    Trainer(gpus=8, strategy="ddp")

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

----------

Jupyter Notebooks
-----------------
Unfortunately any `ddp_` is not supported in jupyter notebooks. Please use `dp` for multiple GPUs. This is a known
Jupyter issue. If you feel like taking a stab at adding this support, feel free to submit a PR!

----------

Pickle Errors
--------------
Multi-GPU training sometimes requires your model to be pickled. If you run into an issue with pickling
try the following to figure out the issue

.. code-block:: python

    import pickle

    model = YourModel()
    pickle.dumps(model)

However, if you use `ddp` the pickling requirement is not there and you should be fine. If you use `ddp_spawn` the
pickling requirement remains. This is a limitation of Python.
