"""
Lightning supports most popular logging frameworks (Tensorboard, comet, weights and biases, etc...).
To use a logger, simply pass it into the trainer. To use multiple loggers, simply pass in a ``list``
or ``tuple`` of loggers.

.. code-block:: python

    from pytorch_lightning import loggers

    # lightning uses tensorboard by default
    tb_logger = loggers.TensorBoardLogger()
    trainer = Trainer(logger=tb_logger)

    # or choose from any of the others such as MLFlow, Comet, Neptune, Wandb
    comet_logger = loggers.CometLogger()
    trainer = Trainer(logger=comet_logger)

    # or pass a list
    tb_logger = loggers.TensorBoardLogger()
    comet_logger = loggers.CometLogger()
    trainer = Trainer(logger=[tb_logger, comet_logger])

.. note:: All loggers log by default to ``os.getcwd()``. To change the path without creating a logger set
    ``Trainer(default_root_dir='/your/path/to/save/checkpoints')``

Custom logger
-------------

You can implement your own logger by writing a class that inherits from
``LightningLoggerBase``. Use the ``rank_zero_only`` decorator to make sure that
only the first process in DDP training logs data.

.. code-block:: python

    from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only

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


If you write a logger that may be useful to others, please send
a pull request to add it to Lighting!

Using loggers
-------------

Call the logger anywhere except ``__init__`` in your LightningModule by doing:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_method_summary_writer_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.add_histogram(...)

Read more in the `Experiment Logging use case <./experiment_logging.html>`_.

Supported Loggers
-----------------
"""
from os import environ

from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection, rank_zero_only
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

__all__ = ['TensorBoardLogger']

try:
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

    from pytorch_lightning.loggers.comet import CometLogger
except ImportError:  # pragma: no-cover
    del environ["COMET_DISABLE_AUTO_LOGGING"]  # pragma: no-cover
else:
    __all__.append('CometLogger')

try:
    from pytorch_lightning.loggers.mlflow import MLFlowLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('MLFlowLogger')

try:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('NeptuneLogger')

try:
    from pytorch_lightning.loggers.test_tube import TestTubeLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('TestTubeLogger')

try:
    from pytorch_lightning.loggers.wandb import WandbLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('WandbLogger')

try:
    from pytorch_lightning.loggers.trains import TrainsLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('TrainsLogger')
