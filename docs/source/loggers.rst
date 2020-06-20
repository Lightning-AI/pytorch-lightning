.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning import loggers as pl_loggers

.. role:: hidden
    :class: hidden-section

Loggers
===========
Lightning supports the most popular logging frameworks (TensorBoard, Comet, Weights and Biases, etc...).
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

Note:
    All loggers log by default to ``os.getcwd()``. To change the path without creating a logger set
    ``Trainer(default_root_dir='/your/path/to/save/checkpoints')``

----------

Custom Logger
-------------

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
            pass

        @rank_zero_only
        def finalize(self, status):
            # Optional. Any code that needs to be run after training
            # finishes goes here
            pass

If you write a logger that may be useful to others, please send
a pull request to add it to Lighting!

----------

Using loggers
-------------

Call the logger anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule` by doing:

.. testcode::

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            # example
            self.logger.experiment.whatever_method_summary_writer_supports(...)

            # example if logger is a tensorboard logger
            self.logger.experiment.add_image('images', grid, 0)
            self.logger.experiment.add_graph(model, images)

        def any_lightning_module_function_or_hook(self):
            self.logger.experiment.add_histogram(...)

Read more in the `Experiment Logging use case <./experiment_logging.html>`_.

------

Supported Loggers
-----------------
The following are loggers we support

Comet
^^^^^

.. autoclass:: pytorch_lightning.loggers.comet.CometLogger
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

Trains
^^^^^^

.. autoclass:: pytorch_lightning.loggers.trains.TrainsLogger
    :noindex: