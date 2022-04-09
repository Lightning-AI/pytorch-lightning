.. testsetup:: *

    import torch
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _gpu:

GPU training (Basic)
====================


What is a GPU?
--------------
A Graphics Processing Unit (GPU), is a specialized hardware accelerator designed to speed up mathematical computations used in gaming and deep learning.

----

Train on 1 GPU
--------------

Make sure you're running on a machine with at least one GPU. There's no need to specify any NVIDIA flags
as Lightning will do it for you.

.. testcode::
    :skipif: torch.cuda.device_count() < 1

    trainer = Trainer(accelerator="gpu", devices=1)

----------------

Train on multiple GPUs
----------------------

To use multiple GPUs, set the number of devices in the Trainer or the index of the GPUs.

.. code::

    trainer = Trainer(accelerator="gpu", devices=4)

Choosing GPU devices
^^^^^^^^^^^^^^^^^^^^

You can select the GPU devices using ranges, a list of indices or a string containing
a comma separated list of GPU ids:

.. testsetup::

    k = 1

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # DEFAULT (int) specifies how many GPUs to use per node
    Trainer(accelerator="gpu", devices=k)

    # Above is equivalent to
    Trainer(accelerator="gpu", devices=list(range(k)))

    # Specify which GPUs to use (don't use when running on cluster)
    Trainer(accelerator="gpu", devices=[0, 1])

    # Equivalent using a string
    Trainer(accelerator="gpu", devices="0, 1")

    # To use all available GPUs put -1 or '-1'
    # equivalent to list(range(torch.cuda.device_count()))
    Trainer(accelerator="gpu", devices=-1)

The table below lists examples of possible input formats and how they are interpreted by Lightning.

+------------------+-----------+---------------------+---------------------------------+
| `devices`        | Type      | Parsed              | Meaning                         |
+==================+===========+=====================+=================================+
| 3                | int       | [0, 1, 2]           | first 3 GPUs                    |
+------------------+-----------+---------------------+---------------------------------+
| -1               | int       | [0, 1, 2, ...]      | all available GPUs              |
+------------------+-----------+---------------------+---------------------------------+
| [0]              | list      | [0]                 | GPU 0                           |
+------------------+-----------+---------------------+---------------------------------+
| [1, 3]           | list      | [1, 3]              | GPUs 1 and 3                    |
+------------------+-----------+---------------------+---------------------------------+
| "3"              | str       | [0, 1, 2]           | first 3 GPUs                    |
+------------------+-----------+---------------------+---------------------------------+
| "1, 3"           | str       | [1, 3]              | GPUs 1 and 3                    |
+------------------+-----------+---------------------+---------------------------------+
| "-1"             | str       | [0, 1, 2, ...]      | all available GPUs              |
+------------------+-----------+---------------------+---------------------------------+

.. note::

    When specifying number of ``devices`` as an integer ``devices=k``, setting the trainer flag
    ``auto_select_gpus=True`` will automatically help you find ``k`` GPUs that are not
    occupied by other processes. This is especially useful when GPUs are configured
    to be in "exclusive mode", such that only one process at a time can access them.
    For more details see the :doc:`trainer guide <../common/trainer>`.

----------

Prepare code for GPU training
-----------------------------
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

FAQ
---

----

How should I adjust the learning rate when using multiple devices?

When using distributed training make sure to modify your learning rate according to your effective
batch size.

Let's say you have a batch size of 7 in your dataloader.

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):
            return Dataset(..., batch_size=7)

In DDP, DDP_SPAWN, Deepspeed, DDP_SHARDED, or Horovod your effective batch size will be 7 * devices * num_nodes.

.. code-block:: python

    # effective batch size = 7 * 8
    Trainer(accelerator="gpu", devices=8, strategy="ddp")
    Trainer(accelerator="gpu", devices=8, strategy="ddp_spawn")
    Trainer(accelerator="gpu", devices=8, strategy="ddp_sharded")
    Trainer(accelerator="gpu", devices=8, strategy="horovod")

    # effective batch size = 7 * 8 * 10
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp_spawn")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp_sharded")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="horovod")

In DDP2 or DP, your effective batch size will be 7 * num_nodes.
The reason is that the full batch is visible to all GPUs on the node when using DDP2.

.. code-block:: python

    # effective batch size = 7
    Trainer(accelerator="gpu", devices=8, strategy="ddp2")
    Trainer(accelerator="gpu", devices=8, strategy="dp")

    # effective batch size = 7 * 10
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp2")
    Trainer(accelerator="gpu", devices=8, strategy="dp")


.. note:: Huge batch sizes are actually really bad for convergence. Check out:
        `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_

----

How do I use multiple GPUs on Jupyter or Colab notebooks?

To use multiple GPUs on notebooks, use the *DP* mode. 

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="dp")

If you want to use other models, please launch your training via the command-shell.

----

I'm getting errors related to Pickling. What do I do?

Pickle is Python's mechanism for serializing and unserializing data. A majority of distributed modes require that your code is fully pickle compliant. If you run into an issue with pickling try the following to figure out the issue

.. code-block:: python

    import pickle

    model = YourModel()
    pickle.dumps(model)

If you `ddp` your code doesn't need to be pickled.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp")

If you use `ddp_spawn` the pickling requirement remains. This is a limitation of Python.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp_spawn")
