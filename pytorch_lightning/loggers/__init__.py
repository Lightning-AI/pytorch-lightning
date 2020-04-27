"""
Lightning supports the most popular logging frameworks (TensorBoard, Comet, Weights and Biases, etc...).
To use a logger, simply pass it into the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
Lightning uses TensorBoard by default.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers
    tb_logger = loggers.TensorBoardLogger('logs/')
    trainer = Trainer(logger=tb_logger)

Choose from any of the others such as MLflow, Comet, Neptune, WandB, ...

.. code-block:: python

    comet_logger = loggers.CometLogger(save_dir='logs/')
    trainer = Trainer(logger=comet_logger)

To use multiple loggers, simply pass in a ``list`` or ``tuple`` of loggers ...

.. code-block:: python

    tb_logger = loggers.TensorBoardLogger('logs/')
    comet_logger = loggers.CometLogger(save_dir='logs/')
    trainer = Trainer(logger=[tb_logger, comet_logger])

Note:
    All loggers log by default to ``os.getcwd()``. To change the path without creating a logger set
    ``Trainer(default_root_dir='/your/path/to/save/checkpoints')``

Custom Logger
-------------

You can implement your own logger by writing a class that inherits from
:class:`LightningLoggerBase`. Use the :func:`~pytorch_lightning.loggers.base.rank_zero_only`
decorator to make sure that only the first process in DDP training logs data.

.. code-block:: python

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

Using loggers
-------------

Call the logger anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule` by doing:

.. code-block:: python

    from pytorch_lightning import LightningModule
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

Supported Loggers
-----------------
"""
from os import environ

from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

__all__ = [
    'LightningLoggerBase',
    'LoggerCollection',
    'TensorBoardLogger',
]

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
