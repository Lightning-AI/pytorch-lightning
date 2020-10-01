.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning import loggers as pl_loggers

.. role:: hidden
    :class: hidden-section
    
.. _loggers:

Loggers
===========
Lightning supports the most popular logging frameworks (TensorBoard, Comet, etc...).
To use a logger, simply pass it into the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
Lightning uses TensorBoard by default.

.. testcode::

    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(logger=tb_logger)

Choose from any of the others such as MLflow, Comet, Neptune, WandB, ...

.. testcode::

    comet_logger = pl_loggers.CometLogger(save_dir='logs/')
    trainer = Trainer(logger=comet_logger)

To use multiple loggers, simply pass in a ``list`` or ``tuple`` of loggers ...

.. testcode::

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    comet_logger = pl_loggers.CometLogger(save_dir='logs/')
    trainer = Trainer(logger=[tb_logger, comet_logger])

.. note::

    All loggers log by default to `os.getcwd()`. To change the path without creating a logger set
    `Trainer(default_root_dir='/your/path/to/save/checkpoints')`

----------

Logging from a LightningModule
------------------------------
Interact with loggers in two ways, automatically and/or manually.

Automatic logging
^^^^^^^^^^^^^^^^^
Use the :func:`~~pytorch_lightning.core.lightning.LightningModule.log` method to log from anywhere in a LightningModule.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log('my_metric', x)

The :func:`~~pytorch_lightning.core.lightning.LightningModule.log` method has a few options:

- on_step (logs the metric at that step in training)
- on_epoch (automatically accumulates and logs at the end of the epoch)
- prog_bar (logs to the progress bar)
- logger (logs to the logger like Tensorboard)

Depending on where log is called from, Lightning auto-determines the correct mode for you. But of course
you can override the default behavior by manually setting the flags

.. note:: Setting on_epoch=True will accumulate your logged values over the full training epoch.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

Once your training starts, you can view the logs by using your favorite logger or booting up the Tensorboard logs:

.. code-block:: bash

    tensorboard --logdir ./lightning_logs


Manual logging
^^^^^^^^^^^^^^
For certain things like histograms, text, images, etc... you may need to use the logger object directly.

.. code-block:: python

    def training_step(...):
        ...
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        tensorboard.add_histogram(...)
        tensorboard.add_figure(...)

----------

Logging from a Callback
-----------------------
To log from a callback, the :func:`~~pytorch_lightning.core.lightning.LightningModule.log`
method of the LightningModule.

.. code-block:: python

    class MyCallback(Callback):

        def on_train_epoch_end(self, trainer, pl_module):
            pl_module.log('something', x)

or access the logger object directly

.. code-block:: python

    class MyCallback(Callback):

        def on_train_epoch_end(self, trainer, pl_module):
            tensorboard = pl_module.logger.experiment
            tensorboard.add_histogram(...)
            tensorboard.add_figure(...)

----------

Make a Custom Logger
--------------------

You can implement your own logger by writing a class that inherits from
:class:`LightningLoggerBase`. Use the :func:`~pytorch_lightning.loggers.base.rank_zero_only`
decorator to make sure that only the first process in DDP training logs data.

.. testcode::

    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.loggers import LightningLoggerBase

    class MyLogger(LightningLoggerBase):

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

Supported Loggers
-----------------
The following are loggers we support

Comet
^^^^^

.. autoclass:: pytorch_lightning.loggers.comet.CometLogger
    :noindex:

CSVLogger
^^^^^^^^^

.. autoclass:: pytorch_lightning.loggers.csv_logs.CSVLogger
    :noindex:

MLFlow
^^^^^^

.. autoclass:: pytorch_lightning.loggers.mlflow.MLFlowLogger
    :noindex:

Neptune
^^^^^^^

.. autoclass:: pytorch_lightning.loggers.neptune.NeptuneLogger
    :noindex:

Tensorboard
^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.loggers.tensorboard.TensorBoardLogger
    :noindex:

Test-tube
^^^^^^^^^

.. autoclass:: pytorch_lightning.loggers.test_tube.TestTubeLogger
    :noindex:

Weights and Biases
^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.loggers.wandb.WandbLogger
    :noindex:
