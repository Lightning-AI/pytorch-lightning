.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning import loggers as pl_loggers

.. role:: hidden
    :class: hidden-section

.. _logging:


#######
Logging
#######

*****************
Supported Loggers
*****************

The following are loggers we support:

.. currentmodule:: pytorch_lightning.loggers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    CometLogger
    CSVLogger
    MLFlowLogger
    NeptuneLogger
    TensorBoardLogger
    WandbLogger


The above loggers will normally plot an additional chart (**global_step VS epoch**). Depending on the loggers you use, there might be some additional charts too.

By default, Lightning uses ``TensorBoard`` logger under the hood, and stores the logs to a directory (by default in ``lightning_logs/``).

.. testcode::

    from pytorch_lightning import Trainer

    # Automatically logs to a directory (by default ``lightning_logs/``)
    trainer = Trainer()

To see your logs:

.. code-block:: bash

    tensorboard --logdir=lightning_logs/

To visualize tensorboard in a jupyter notebook environment, run the following command in a jupyter cell:

.. code-block:: bash

    %reload_ext tensorboard
    %tensorboard --logdir=lightning_logs/

You can also pass a custom Logger to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

.. testcode::

    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = Trainer(logger=tb_logger)

Choose from any of the others such as MLflow, Comet, Neptune, WandB, etc.

.. testcode::

    comet_logger = pl_loggers.CometLogger(save_dir="logs/")
    trainer = Trainer(logger=comet_logger)

To use multiple loggers, simply pass in a ``list`` or ``tuple`` of loggers.

.. testcode::

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    comet_logger = pl_loggers.CometLogger(save_dir="logs/")
    trainer = Trainer(logger=[tb_logger, comet_logger])

.. note::

    By default, Lightning logs every 50 steps. Use Trainer flags to :ref:`logging_frequency`.

.. note::

    By default, all loggers log to ``os.getcwd()``. You can change the logging path using
    ``Trainer(default_root_dir="/your/path/to/save/checkpoints")`` without instantiating a logger.

----------

******************************
Logging from a LightningModule
******************************

Lightning offers automatic log functionalities for logging scalars, or manual logging for anything else.

Automatic Logging
=================

Use the :meth:`~pytorch_lightning.core.lightning.LightningModule.log`
method to log from anywhere in a :doc:`lightning module <../common/lightning_module>` and :doc:`callbacks <../extensions/callbacks>`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_metric", x)


    # or a dict to get multiple metrics on the same plot if the logger supports it
    def training_step(self, batch, batch_idx):
        self.log("performance", {"acc": acc, "recall": recall})

Depending on where the :meth:`~pytorch_lightning.core.lightning.LightningModule.log` method is called, Lightning auto-determines
the correct logging mode for you. Of course you can override the default behavior by manually setting the
:meth:`~pytorch_lightning.core.lightning.LightningModule.log` parameters.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


Manual Logging Non-Scalar Artifacts
===================================

If you want to log anything that is not a scalar, like histograms, text, images, etc., you may need to use the logger object directly.

.. code-block:: python

    def training_step(self):
        ...
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        tensorboard.add_image()
        tensorboard.add_histogram(...)
        tensorboard.add_figure(...)


----------

********************
Make a Custom Logger
********************

You can implement your own logger by writing a class that inherits from :class:`~pytorch_lightning.loggers.base.LightningLoggerBase`.
Use the :func:`~pytorch_lightning.loggers.base.rank_zero_experiment` and :func:`~pytorch_lightning.utilities.rank_zero.rank_zero_only` decorators to make sure that only the first process in DDP training creates the experiment and logs the data respectively.

.. testcode::

    from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
    from pytorch_lightning.utilities.distributed import rank_zero_only


    class MyLogger(LightningLoggerBase):
        @property
        def name(self):
            return "MyLogger"

        @property
        def version(self):
            # Return the experiment version, int or str.
            return "0.1"

        @rank_zero_only
        def log_hyperparams(self, params):
            # params is an argparse.Namespace
            # your code to record hyperparameters goes here
            pass

        @rank_zero_only
        def log_metrics(self, metrics, step):
            # metrics is a dictionary of metric names and values
            # your code to record metrics goes here
            pass

        @rank_zero_only
        def save(self):
            # Optional. Any code necessary to save logger data goes here
            pass

        @rank_zero_only
        def finalize(self, status):
            # Optional. Any code that needs to be run after training
            # finishes goes here
            pass

If you write a logger that may be useful to others, please send
a pull request to add it to Lightning!

----------

.. _logging_frequency:


*************************
Control Logging Frequency
*************************

Logging frequency
=================

It may slow down training to log on every single batch. By default, Lightning logs every 50 rows, or 50 training steps.
To change this behaviour, set the ``log_every_n_steps`` :class:`~pytorch_lightning.trainer.trainer.Trainer` flag.

.. testcode::

   k = 10
   trainer = Trainer(log_every_n_steps=k)


Log Writing Frequency
=====================

Individual logger implementations determine their flushing frequency. For example, on the
:class:`~pytorch_lightning.loggers.csv_logs.CSVLogger` you can set the flag ``flush_logs_every_n_steps``.

----------

************
Progress Bar
************

You can add any metric to the progress bar using :meth:`~pytorch_lightning.core.lightning.LightningModule.log`
method, setting ``prog_bar=True``.


.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_loss", loss, prog_bar=True)


You could learn more about progress bars supported by Lightning :doc:`here <../common/progress_bar>`.

Modifying the Progress Bar
==========================

The progress bar by default already includes the training loss and version number of the experiment
if you are using a logger. These defaults can be customized by overriding the
:meth:`~pytorch_lightning.callbacks.progress.base.ProgressBarBase.get_metrics` hook in your logger.

.. code-block:: python

    from pytorch_lightning.callbacks.progress import Tqdm


    class CustomProgressBar(Tqdm):
        def get_metrics(self, *args, **kwargs):
            # don't show the version number
            items = super().get_metrics()
            items.pop("v_num", None)
            return items


----------


*************************
Configure Console Logging
*************************

Lightning logs useful information about the training process and user warnings to the console.
You can retrieve the Lightning console logger and change it to your liking. For example, adjust the logging level
or redirect output for certain modules to log files:

.. testcode::

    import logging

    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler("core.log"))

Read more about custom Python logging `here <https://docs.python.org/3/library/logging.html>`_.


----------

***********************
Logging Hyperparameters
***********************

When training a model, it is useful to know what hyperparams went into that model.
When Lightning creates a checkpoint, it stores a key ``"hyper_parameters"`` with the hyperparams.

.. code-block:: python

    lightning_checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    hyperparams = lightning_checkpoint["hyper_parameters"]

Some loggers also allow logging the hyperparams used in the experiment. For instance,
when using the ``TensorBoardLogger``, all hyperparams will show
in the `hparams tab <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>`_.

.. note::
    If you want to track a metric in the tensorboard hparams tab, log scalars to the key ``hp_metric``. If tracking multiple metrics, initialize ``TensorBoardLogger`` with ``default_hp_metric=False`` and call ``log_hyperparams`` only once with your metric keys and initial values. Subsequent updates can simply be logged to the metric keys. Refer to the examples below for setting up proper hyperparams metrics tracking within the :doc:`LightningModule <../common/lightning_module>`.

    .. code-block:: python

        # Using default_hp_metric
        def validation_step(self, batch, batch_idx):
            self.log("hp_metric", some_scalar)


        # Using custom or multiple metrics (default_hp_metric=False)
        def on_train_start(self):
            self.logger.log_hyperparams(self.hparams, {"hp/metric_1": 0, "hp/metric_2": 0})


        def validation_step(self, batch, batch_idx):
            self.log("hp/metric_1", some_scalar_1)
            self.log("hp/metric_2", some_scalar_2)

    In the example, using ``"hp/"`` as a prefix allows for the metrics to be grouped under "hp" in the tensorboard scalar tab where you can collapse them.

-----------

***************************
Managing Remote Filesystems
***************************

Lightning supports saving logs to a variety of filesystems, including local filesystems and several cloud storage providers.

Check out the :ref:`Remote Filesystems <remote_fs>` doc for more info.
