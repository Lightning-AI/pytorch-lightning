:orphan:

.. _logging_advanced:

##########################################
Track and Visualize Experiments (advanced)
##########################################
**Audience:** Users who want to do advanced speed optimizations by customizing the logging behavior.

----

****************************
Change progress bar defaults
****************************
To change the default values (ie: version number) shown in the progress bar, override the :meth:`~lightning.pytorch.callbacks.progress.progress_bar.ProgressBar.get_metrics` method in your logger.

.. code-block:: python

    from lightning.pytorch.callbacks.progress import Tqdm


    class CustomProgressBar(Tqdm):
        def get_metrics(self, *args, **kwargs):
            # don't show the version number
            items = super().get_metrics()
            items.pop("v_num", None)
            return items

----

************************************
Customize tracking to speed up model
************************************


Modify logging frequency
========================

Logging a metric on every single batch can slow down training. By default, Lightning logs every 50 rows, or 50 training steps.
To change this behaviour, set the *log_every_n_steps* :class:`~lightning.pytorch.trainer.trainer.Trainer` flag.

.. testcode::

   k = 10
   trainer = Trainer(log_every_n_steps=k)

----

Modify flushing frequency
=========================

Some loggers keep logged metrics in memory for N steps and only periodically flush them to disk to improve training efficiency.
Every logger handles this a bit differently. For example, here is how to fine-tune flushing for the TensorBoard logger:

.. code-block:: python

    # Default used by TensorBoard: Write to disk after 10 logging events or every two minutes
    logger = TensorBoardLogger(..., max_queue=10, flush_secs=120)

    # Faster training, more memory used
    logger = TensorBoardLogger(..., max_queue=100)

    # Slower training, less memory used
    logger = TensorBoardLogger(..., max_queue=1)

----

******************
Customize self.log
******************

The LightningModule *self.log* method offers many configurations to customize its behavior.

----

add_dataloader_idx
==================
**Default:** True

If True, appends the index of the current dataloader to the name (when using multiple dataloaders). If False, user needs to give unique names for each dataloader to not mix the values.

.. code-block:: python

  self.log(add_dataloader_idx=True)

----

batch_size
==========
**Default:** None

Current batch size used for accumulating logs logged with ``on_epoch=True``. This will be directly inferred from the loaded batch, but for some data structures you might need to explicitly provide it.

.. code-block:: python

  self.log(batch_size=32)

----

enable_graph
============
**Default:** True

If True, will not auto detach the graph.

.. code-block:: python

  self.log(enable_graph=True)

----

logger
======
**Default:** True

Send logs to the logger like ``Tensorboard``, or any other custom logger passed to the :class:`~lightning.pytorch.trainer.trainer.Trainer` (Default: ``True``).

.. code-block:: python

  self.log(logger=True)

----

on_epoch
========
**Default:** It varies

If this is True, that specific *self.log* call accumulates and reduces all metrics to the end of the epoch.

.. code-block:: python

  self.log(on_epoch=True)

The default value depends in which function this is called

.. code-block:: python

  def training_step(self, batch, batch_idx):
      # Default: False
      self.log(on_epoch=False)


  def validation_step(self, batch, batch_idx):
      # Default: True
      self.log(on_epoch=True)


  def test_step(self, batch, batch_idx):
      # Default: True
      self.log(on_epoch=True)

----

on_step
=======
**Default:** It varies

If this is True, that specific *self.log* call will NOT accumulate metrics. Instead it will generate a timeseries across steps.

.. code-block:: python

  self.log(on_step=True)

The default value depends in which function this is called

.. code-block:: python

  def training_step(self, batch, batch_idx):
      # Default: True
      self.log(on_step=True)


  def validation_step(self, batch, batch_idx):
      # Default: False
      self.log(on_step=False)


  def test_step(self, batch, batch_idx):
      # Default: False
      self.log(on_step=False)


----

prog_bar
========
**Default:** False

If set to True, logs will be sent to the progress bar.

.. code-block:: python

  self.log(prog_bar=True)

----

rank_zero_only
==============
**Default:** False

Tells Lightning if you are calling ``self.log`` from every process (default) or only from rank 0.
This is for advanced users who want to reduce their metric manually across processes, but still want to benefit from automatic logging via ``self.log``.

- Set ``False`` (default) if you are calling ``self.log`` from every process.
- Set ``True`` if you are calling ``self.log`` from rank 0 only. Caveat: you won't be able to use this metric as a monitor in callbacks (e.g., early stopping).

.. code-block:: python

    # Default
    self.log(..., rank_zero_only=False)

    # If you call `self.log` on rank 0 only, you need to set `rank_zero_only=True`
    if self.trainer.global_rank == 0:
        self.log(..., rank_zero_only=True)

    # DON'T do this, it will cause deadlocks!
    self.log(..., rank_zero_only=True)


----

reduce_fx
=========
**Default:** :func:`torch.mean`

Reduction function over step values for end of epoch. Uses :func:`torch.mean` by default and is not applied when a :class:`torchmetrics.Metric` is logged.

.. code-block:: python

  self.log(..., reduce_fx=torch.mean)

----

sync_dist
=========
**Default:** False

If True, reduces the metric across devices. Use with care as this may lead to a significant communication overhead.

.. code-block:: python

  self.log(sync_dist=False)

----

sync_dist_group
===============
**Default:** None

The DDP group to sync across.

.. code-block:: python

  import torch.distributed as dist

  group = dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)
  self.log(sync_dist_group=group)

----

***************************************
Enable metrics for distributed training
***************************************
For certain types of metrics that need complex aggregation, we recommended to build your metric using torchmetric which ensures all the complexities of metric aggregation in distributed environments is handled.

First, implement your metric:

.. code-block:: python

  import torch
  import torchmetrics


  class MyAccuracy(Metric):
      def __init__(self, dist_sync_on_step=False):
          # call `self.add_state`for every internal state that is needed for the metrics computations
          # dist_reduce_fx indicates the function that should be used to reduce
          # state from multiple processes
          super().__init__(dist_sync_on_step=dist_sync_on_step)

          self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
          self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

      def update(self, preds: torch.Tensor, target: torch.Tensor):
          # update metric states
          preds, target = self._input_format(preds, target)
          assert preds.shape == target.shape

          self.correct += torch.sum(preds == target)
          self.total += target.numel()

      def compute(self):
          # compute final result
          return self.correct.float() / self.total

To use the metric inside Lightning, 1) initialize it in the init, 2) compute the metric, 3) pass it into *self.log*

.. code-block:: python

  class LitModel(LightningModule):
      def __init__(self):
          # 1. initialize the metric
          self.accuracy = MyAccuracy()

      def training_step(self, batch, batch_idx):
          x, y = batch
          preds = self(x)

          # 2. compute the metric
          self.accuracy(preds, y)

          # 3. log it
          self.log("train_acc_step", self.accuracy)

----

********************************
Log to a custom cloud filesystem
********************************
Lightning is integrated with the major remote file systems including local filesystems and several cloud storage providers such as
`S3 <https://aws.amazon.com/s3/>`_ on `AWS <https://aws.amazon.com/>`_, `GCS <https://cloud.google.com/storage>`_ on `Google Cloud <https://cloud.google.com/>`_,
or `ADL <https://azure.microsoft.com/solutions/data-lake/>`_ on `Azure <https://azure.microsoft.com/>`_.

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.

To save logs to a remote filesystem, prepend a protocol like "s3:/" to the root_dir used for writing and reading model data.

.. code-block:: python

    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="s3://my_bucket/logs/")

    trainer = Trainer(logger=logger)
    trainer.fit(model)

----

*********************************
Track both step and epoch metrics
*********************************
To track the timeseries over steps (*on_step*) as well as the accumulated epoch metric (*on_epoch*), set both to True

.. code-block:: python

  self.log(on_step=True, on_epoch=True)

Setting both to True will generate two graphs with *_step* for the timeseries over steps and *_epoch* for the epoch metric.

.. TODO:: show images of both

----

**************************************
Understand self.log automatic behavior
**************************************
This table shows the default values of *on_step* and *on_epoch* depending on the *LightningModule* or *Callback* method.

----

In LightningModule
==================

.. list-table:: Default behavior of logging in ightningModule
   :widths: 50 25 25
   :header-rows: 1

   * - Method
     - on_step
     - on_epoch
   * - on_after_backward, on_before_backward, on_before_optimizer_step, optimizer_step, configure_gradient_clipping, on_before_zero_grad, training_step
     - True
     - False
   * - test_step, validation_step
     - False
     - True

----

In Callback
===========

.. list-table:: Default behavior of logging in Callback
   :widths: 50 25 25
   :header-rows: 1

   * - Method
     - on_step
     - on_epoch
   * - on_after_backward, on_before_backward, on_before_optimizer_step, on_before_zero_grad, on_train_batch_start, on_train_batch_end
     - True
     - False
   * - on_train_epoch_start, on_train_epoch_end, on_train_start, on_validation_batch_start, on_validation_batch_end, on_validation_start, on_validation_epoch_start, on_validation_epoch_end
     - False
     - True

.. note:: To add logging to an unsupported method, please open an issue with a clear description of why it is blocking you.
