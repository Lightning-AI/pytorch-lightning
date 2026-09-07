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

When running in distributed mode, each rank runs ``validation_step`` and ``test_step`` on its own
shard of the data. Without explicit synchronization, the value your logger persists is rank 0's
local value — computed on just ``1 / world_size`` of the validation or test set. That is the
metric your :class:`~lightning.pytorch.callbacks.ModelCheckpoint` and
:class:`~lightning.pytorch.callbacks.EarlyStopping` callbacks see, so an unsynchronized metric
can silently pick the wrong checkpoint.

Lightning gives you three tools to fix this, and they are **not interchangeable**:

- ``sync_dist=True`` — mean-reduces a scalar across ranks. Correct only for averageable metrics.
- `TorchMetrics <https://torchmetrics.readthedocs.io/>`__ — syncs the metric's internal *state*, then computes. Correct for non-averageable metrics such as F1 or AUC.
- :meth:`~lightning.pytorch.core.LightningModule.all_gather` — gathers raw tensors across ranks so you can compute any reduction yourself.

Pick the lightest tool that fits the metric. If you accumulate per-step outputs and compute a
custom metric in ``on_validation_epoch_end`` (or ``on_test_epoch_end``), jump to
:ref:`manual-all-gather` — that is the pattern most DDP custom-metric questions come down to.

``sync_dist=True``
==================

The simplest option. Lightning mean-reduces each logged value across all ranks before storing it.

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

The ``sync_dist`` option can also be used in logging calls during the training step, but be aware
that this can lead to significant communication overhead and slow down your training.

.. warning::
    ``sync_dist=True`` averages per-rank *values*. It is only correct when
    ``mean(per_rank_metric) == global_metric``. It is **wrong** for F1, AUC, and precision or
    recall on imbalanced classes — the mean of per-rank F1 scores is not the global F1 score.
    For those metrics, reach for TorchMetrics instead.

TorchMetrics
============

`TorchMetrics <https://torchmetrics.readthedocs.io/>`__ handles the non-averageable case by
syncing the metric's internal *state* (for example, the running counts of true and false
positives) across ranks, then computing the metric from the merged state. The result matches
what you would get by evaluating on one rank with the full dataset. No ``sync_dist`` flag is
needed; the metric synchronizes itself when it is logged.

.. code-block:: python

    from torchmetrics.classification import BinaryF1Score


    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.val_f1 = BinaryF1Score()


        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            self.val_f1.update(logits, y)
            # Passing the metric object to self.log triggers DDP sync at epoch end.
            self.log("val_f1", self.val_f1, on_epoch=True)

This is the recommended option for any classification, retrieval, or ranking metric.

.. _manual-all-gather:

Manual ``all_gather``
=====================

Use this when your metric is a custom computation over outputs accumulated across the whole
epoch — the case where neither ``sync_dist`` nor TorchMetrics fits. The pattern is: accumulate
per-step outputs into a list on the module, then at epoch end call
:meth:`~lightning.pytorch.core.LightningModule.all_gather` to combine each rank's contributions
before computing the metric. ``all_gather`` returns a tensor of shape
``[world_size, *tensor_shape]`` and every rank receives the same result.

.. code-block:: python

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.val_outputs = []


        def validation_step(self, batch, batch_idx):
            x, y = batch
            predictions = self(x)
            self.val_outputs.append(predictions)
            return predictions


        def on_validation_epoch_end(self):
            # self.all_gather returns a tensor of shape [world_size, *tensor_shape] on every rank.
            gathered = self.all_gather(self.val_outputs)
            metric = my_custom_metric(gathered)
            self.val_outputs.clear()  # free memory before the next epoch

            # When you call `self.log` only on rank 0, don't forget to add
            # `rank_zero_only=True` to avoid deadlocks on synchronization.
            # Caveat: monitoring this is unimplemented, see https://github.com/Lightning-AI/pytorch-lightning/issues/15852
            if self.trainer.is_global_zero:
                self.log("my_custom_val_metric", metric, rank_zero_only=True)

The same pattern applies to ``test_step`` / ``on_test_epoch_end``.

A common source of confusion here is that ``on_validation_epoch_end`` runs on every rank, so at
first glance the metric looks like it is being computed ``world_size`` times. After
``all_gather`` every rank already holds the *same* gathered tensor, so every rank computes the
*same* value — the redundant work is cheap and the result is correct. The ``is_global_zero``
guard belongs around ``self.log``, not around the computation. Never guard ``all_gather``
itself with ``is_global_zero``: it is a collective, and if some ranks skip it the program will
hang.

Which one should I use?
=======================

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Metric
      - Use
    * - Averageable scalar (loss, accuracy, MSE)
      - ``sync_dist=True``
    * - Classification or ranking metric (F1, AUC, precision, recall)
      - TorchMetrics
    * - Custom reduction over gathered tensors
      - ``self.all_gather()``

Common pitfalls
===============

- **Using** ``sync_dist=True`` **on a non-averageable metric.** The logged value is the mean of
  per-rank metrics, which is not the global metric. Use TorchMetrics instead.
- **Guarding** ``all_gather`` **with** ``is_global_zero``. Collectives must be called on every
  rank. Put the guard around ``self.log``, not around the gather.
- **Passing** ``rank_zero_only=True`` **to** ``self.log`` **without synchronizing first.** Rank 0
  logs its local value only, which is the ``1 / world_size`` problem this section opens with.

See also: the `TorchMetrics distributed evaluation guide
<https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metrics-and-distributed-training-ddp>`_
for how TorchMetrics synchronizes state internally.


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
