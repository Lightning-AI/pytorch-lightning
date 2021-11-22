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

Lightning supports the most popular logging frameworks (TensorBoard, Comet, etc...).

By default, Lightning uses `PyTorch TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`__ logging  under the hood, and stores the logs to a directory (by default in ``lightning_logs/``).

.. testcode::

    from pytorch_lightning import Trainer

    # Automatically logs to a directory
    # (by default ``lightning_logs/``)
    trainer = Trainer()

To see your logs:

.. code-block:: bash

    tensorboard --logdir=lightning_logs/

You can also pass a custom Logger to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

.. testcode::

    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    trainer = Trainer(logger=tb_logger)

Choose from any of the others such as MLflow, Comet, Neptune, WandB, ...

.. testcode::

    comet_logger = pl_loggers.CometLogger(save_dir="logs/")
    trainer = Trainer(logger=comet_logger)

To use multiple loggers, simply pass in a ``list`` or ``tuple`` of loggers ...

.. testcode::

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    comet_logger = pl_loggers.CometLogger(save_dir="logs/")
    trainer = Trainer(logger=[tb_logger, comet_logger])

.. note::

    By default, lightning logs every 50 steps. Use Trainer flags to :ref:`logging_frequency`.

.. note::

    All loggers log by default to `os.getcwd()`. To change the path without creating a logger set
    `Trainer(default_root_dir='/your/path/to/save/checkpoints')`

----------

******************************
Logging from a LightningModule
******************************

Lightning offers automatic log functionalities for logging scalars, or manual logging for anything else.

Automatic Logging
=================
Use the :func:`~~pytorch_lightning.core.lightning.LightningModule.log`
method to log from anywhere in a :doc:`lightning module <../common/lightning_module>` and :doc:`callbacks <../extensions/callbacks>`
except functions with `batch_start` in their names.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_metric", x)


    # or a dict
    def training_step(self, batch, batch_idx):
        self.log("performance", {"acc": acc, "recall": recall})

Depending on where log is called from, Lightning auto-determines the correct logging mode for you. \
But of course you can override the default behavior by manually setting the :func:`~~pytorch_lightning.core.lightning.LightningModule.log` parameters.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

The :func:`~~pytorch_lightning.core.lightning.LightningModule.log` method has a few options:

* `on_step`: Logs the metric at the current step. Defaults to `True` in :func:`~~pytorch_lightning.core.lightning.LightningModule.training_step`, and :func:`~pytorch_lightning.core.lightning.LightningModule.training_step_end`.

* `on_epoch`: Automatically accumulates and logs at the end of the epoch. Defaults to True anywhere in validation or test loops, and in :func:`~~pytorch_lightning.core.lightning.LightningModule.training_epoch_end`.

* `prog_bar`: Logs to the progress bar.

* `logger`: Logs to the logger like Tensorboard, or any other custom logger passed to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.


.. note::

    -   Setting ``on_epoch=True`` will cache all your logged values during the full training epoch and perform a
        reduction in ``on_train_epoch_end``. We recommend using `TorchMetrics <https://torchmetrics.readthedocs.io/>`_, when working with custom reduction.

    -   Setting both ``on_step=True`` and ``on_epoch=True`` will create two keys per metric you log with
        suffix ``_step`` and ``_epoch``, respectively. You can refer to these keys e.g. in the `monitor`
        argument of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` or in the graphs plotted to the logger of your choice.


If your work requires to log in an unsupported function, please open an issue with a clear description of why it is blocking you.


Manual logging
==============
If you want to log anything that is not a scalar, like histograms, text, images, etc... you may need to use the logger object directly.

.. code-block:: python

    def training_step(self):
        ...
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        tensorboard.add_image()
        tensorboard.add_histogram(...)
        tensorboard.add_figure(...)


Access your logs
================
Once your training starts, you can view the logs by using your favorite logger or booting up the Tensorboard logs:

.. code-block:: bash

    tensorboard --logdir ./lightning_logs

----------

********************
Make a custom logger
********************

You can implement your own logger by writing a class that inherits from :class:`~pytorch_lightning.loggers.base.LightningLoggerBase`.
Use the :func:`~pytorch_lightning.loggers.base.rank_zero_experiment` and :func:`~pytorch_lightning.utilities.distributed.rank_zero_only` decorators to make sure that only the first process in DDP training creates the experiment and logs the data respectively.

.. testcode::

    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.loggers import LightningLoggerBase
    from pytorch_lightning.loggers.base import rank_zero_experiment


    class MyLogger(LightningLoggerBase):
        @property
        def name(self):
            return "MyLogger"

        @property
        @rank_zero_experiment
        def experiment(self):
            # Return the experiment object associated with this logger.
            pass

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
            # If you implement this, remember to call `super().save()`
            # at the start of the method (important for aggregation of metrics)
            super().save()

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
Control logging frequency
*************************

Logging frequency
=================

It may slow training down to log every single batch. By default, Lightning logs every 50 rows, or 50 training steps.
To change this behaviour, set the `log_every_n_steps` :class:`~pytorch_lightning.trainer.trainer.Trainer` flag.

.. testcode::

   k = 10
   trainer = Trainer(log_every_n_steps=k)



Log writing frequency
=====================

Writing to a logger can be expensive, so by default Lightning writes logs to disk or to the given logger every 100 training steps.
To change this behaviour, set the interval at which you wish to flush logs to the filesystem using the `flush_logs_every_n_steps` :class:`~pytorch_lightning.trainer.trainer.Trainer` flag.

.. testcode::

    k = 100
    trainer = Trainer(flush_logs_every_n_steps=k)

Unlike the `log_every_n_steps`, this argument does not apply to all loggers.
The example shown here works with :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`,
which is the default logger in Lightning.

----------

************
Progress Bar
************
You can add any metric to the progress bar using :func:`~~pytorch_lightning.core.lightning.LightningModule.log`
method, setting `prog_bar=True`.


.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("my_loss", loss, prog_bar=True)


Modifying the progress bar
==========================

The progress bar by default already includes the training loss and version number of the experiment
if you are using a logger. These defaults can be customized by overriding the
:func:`~pytorch_lightning.callbacks.base.ProgressBarBase.get_metrics` hook in your module.

.. code-block:: python

    def get_metrics(self):
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        return items


----------


*************************
Configure console logging
*************************

Lightning logs useful information about the training process and user warnings to the console.
You can retrieve the Lightning logger and change it to your liking. For example, adjust the logging level
or redirect output for certain modules to log files:

.. testcode::

    import logging

    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler("core.log"))

Read more about custom Python logging `here <https://docs.python.org/3/library/logging.html>`_.


----------

***********************
Logging hyperparameters
***********************

When training a model, it's useful to know what hyperparams went into that model.
When Lightning creates a checkpoint, it stores a key "hyper_parameters" with the hyperparams.

.. code-block:: python

    lightning_checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    hyperparams = lightning_checkpoint["hyper_parameters"]

Some loggers also allow logging the hyperparams used in the experiment. For instance,
when using the TestTubeLogger or the TensorBoardLogger, all hyperparams will show
in the `hparams tab <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>`_.

.. note::
    If you want to track a metric in the tensorboard hparams tab, log scalars to the key ``hp_metric``. If tracking multiple metrics, initialize ``TensorBoardLogger`` with ``default_hp_metric=False`` and call ``log_hyperparams`` only once with your metric keys and initial values. Subsequent updates can simply be logged to the metric keys. Refer to the following for examples on how to setup proper hyperparams metrics tracking within :doc:`LightningModule <../common/lightning_module>`.

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

    In the example, using `hp/` as a prefix allows for the metrics to be grouped under "hp" in the tensorboard scalar tab where you can collapse them.

----------

*************
Snapshot code
*************

Loggers also allow you to snapshot a copy of the code used in this experiment.
For example, TestTubeLogger does this with a flag:

.. code-block:: python

    from pytorch_lightning.loggers import TestTubeLogger

    logger = TestTubeLogger(".", create_git_tag=True)

----------

*****************
Supported Loggers
*****************

The following are loggers we support

.. note::
    The following loggers will normally plot an additional chart (**global_step VS epoch**).

.. note::
    postfix ``_step`` and ``_epoch`` will be appended to the name you logged
    if ``on_step`` and ``on_epoch`` are set to ``True`` in ``self.log()``.

.. note::
    Depending on the loggers you use, there might be some additional charts.

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
    TestTubeLogger
    WandbLogger
