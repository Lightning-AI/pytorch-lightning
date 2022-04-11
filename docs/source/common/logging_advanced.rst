.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _loggers:

########################################
Track and visualize artifacts (advanced)
########################################
A

----

***********************
Change the progress bar
***********************
To change the default values (ie: version number) shown in the progress bar, override the :meth:`~pytorch_lightning.callbacks.progress.base.ProgressBarBase.get_metrics` method in your logger.

.. code-block:: python

    from pytorch_lightning.callbacks.progress import Tqdm

    class CustomProgressBar(Tqdm):
        def get_metrics(self, *args, **kwargs):
            # don't show the version number
            items = super().get_metrics()
            items.pop("v_num", None)
            return items

----

***********************************************
Modify logging frequency to speed up your model
***********************************************


Modify logging frequency
========================

Logging a metric on every single batch can slow down training. By default, Lightning logs every 50 rows, or 50 training steps.
To change this behaviour, set the *log_every_n_steps* :class:`~pytorch_lightning.trainer.trainer.Trainer` flag.

.. testcode::

   k = 10
   trainer = Trainer(log_every_n_steps=k)

----

Modify flushing frequency
=========================

Metrics are kept in memory for N steps to improve training efficiency. Every N steps, metrics flush to disk. To change the frequency of this flushing, use the *flush_logs_every_n_steps* Trainer argument.

.. code-block:: python

    # faster training, high memory
    Trainer(flush_logs_every_n_steps=500)
    
    # slower training, low memory
    Trainer(flush_logs_every_n_steps=500)

The higher *flush_logs_every_n_steps* is, the faster the model will train but the memory will build up until the next flush.
The smaller *flush_logs_every_n_steps* is, the slower the model will train but memory will be kept to a minimum.

TODO: chart

----

******************
Customize self.log 
******************
The :meth:`~pytorch_lightning.core.lightning.LightningModule.log` method has a few options:

* ``on_step``: Logs the metric at the current step.
* ``on_epoch``: Automatically accumulates and logs at the end of the epoch.
* ``prog_bar``: Logs to the progress bar (Default: ``False``).
* ``logger``: Logs to the logger like ``Tensorboard``, or any other custom logger passed to the :class:`~pytorch_lightning.trainer.trainer.Trainer` (Default: ``True``).
* ``reduce_fx``: Reduction function over step values for end of epoch. Uses :meth:`torch.mean` by default.
* ``enable_graph``: If True, will not auto detach the graph.
* ``sync_dist``: If True, reduces the metric across devices. Use with care as this may lead to a significant communication overhead.
* ``sync_dist_group``: The DDP group to sync across.
* ``add_dataloader_idx``: If True, appends the index of the current dataloader to the name (when using multiple dataloaders). If False, user needs to give unique names for each dataloader to not mix the values.
* ``batch_size``: Current batch size used for accumulating logs logged with ``on_epoch=True``. This will be directly inferred from the loaded batch, but for some data structures you might need to explicitly provide it.
* ``rank_zero_only``: Whether the value will be logged only on rank 0. This will prevent synchronization which would produce a deadlock as not all processes would perform this log call.

.. list-table:: Default behavior of logging in Callback or LightningModule
   :widths: 50 25 25
   :header-rows: 1

   * - Hook
     - on_step
     - on_epoch
   * - on_train_start, on_train_epoch_start, on_train_epoch_end, training_epoch_end
     - False
     - True
   * - on_before_backward, on_after_backward, on_before_optimizer_step, on_before_zero_grad
     - True
     - False
   * - on_train_batch_start, on_train_batch_end, training_step, training_step_end
     - True
     - False
   * - on_validation_start, on_validation_epoch_start, on_validation_epoch_end, validation_epoch_end
     - False
     - True
   * - on_validation_batch_start, on_validation_batch_end, validation_step, validation_step_end
     - False
     - True

.. note::

    - The above config for ``validation`` applies for ``test`` hooks as well.

    -   Setting ``on_epoch=True`` will cache all your logged values during the full training epoch and perform a
        reduction in ``on_train_epoch_end``. We recommend using `TorchMetrics <https://torchmetrics.readthedocs.io/>`_, when working with custom reduction.

    -   Setting both ``on_step=True`` and ``on_epoch=True`` will create two keys per metric you log with
        suffix ``_step`` and ``_epoch`` respectively. You can refer to these keys e.g. in the `monitor`
        argument of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` or in the graphs plotted to the logger of your choice.


If your work requires to log in an unsupported method, please open an issue with a clear description of why it is blocking you.

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

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="s3://my_bucket/logs/")

    trainer = Trainer(logger=logger)
    trainer.fit(model)

----

***************************************
Enable metrics for distributed training
***************************************
A