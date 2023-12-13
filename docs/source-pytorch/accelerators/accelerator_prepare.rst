:orphan:

########################################
Hardware agnostic training (preparation)
########################################

To train on CPU/GPU/TPU without changing your code, we need to build a few good habits :)

----

*****************************
Delete .cuda() or .to() calls
*****************************

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

----

************************************************
Init tensors using Tensor.to and register_buffer
************************************************
When you need to create a new tensor, use ``Tensor.to``.
This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.

.. testcode::

    # before lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.cuda(0)


    # with lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.to(x)

The :class:`~lightning.pytorch.core.LightningModule` knows what device it is on. You can access the reference via ``self.device``.
Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will
remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic,
register the tensor as a buffer in your modules' ``__init__`` method with :meth:`~torch.nn.Module.register_buffer`.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            ...
            self.register_buffer("sigma", torch.eye(3))
            # you can now access self.sigma anywhere in your module

----

***************
Remove samplers
***************

:class:`~torch.utils.data.distributed.DistributedSampler` is automatically handled by Lightning.

See :ref:`replace-sampler-ddp` for more information.

----

***************************************
Synchronize validation and test logging
***************************************

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

.. code-block:: python

    def __init__(self):
        super().__init__()
        self.outputs = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        tensors = self(x)
        self.outputs.append(tensors)
        return tensors


    def on_test_epoch_end(self):
        mean = torch.mean(self.all_gather(self.outputs))
        self.outputs.clear()  # free memory

        # When you call `self.log` only on rank 0, don't forget to add
        # `rank_zero_only=True` to avoid deadlocks on synchronization.
        # Caveat: monitoring this is unimplemented, see https://github.com/Lightning-AI/lightning/issues/15852
        if self.trainer.is_global_zero:
            self.log("my_reduced_metric", mean, rank_zero_only=True)


----


**********************
Make models pickleable
**********************
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
