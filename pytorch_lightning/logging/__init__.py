"""
Lighting offers options for logging information about model, gpu usage, etc,
 via several different logging frameworks. It also offers printing options for training monitoring.

**default_save_path**

Lightning sets a default TestTubeLogger and CheckpointCallback for you which log to
`os.getcwd()` by default. To modify the logging path you can set::

    Trainer(default_save_path='/your/path/to/save/checkpoints')


If you need more custom behavior (different paths for both, different metrics, etc...)
 from the logger and the checkpointCallback, pass in your own instances as explained below.

Setting up logging
------------------

The trainer inits a default logger for you (TestTubeLogger). All logs will
go to the current working directory under a folder named `os.getcwd()/lightning_logs`.

If you want to modify the default logging behavior even more, pass in a logger
 (which should inherit from `LightningBaseLogger`).

.. code-block:: python

    my_logger = MyLightningLogger(...)
    trainer = Trainer(logger=my_logger)


The path in this logger will overwrite `default_save_path`.

Lightning supports several common experiment tracking frameworks out of the box

Custom logger
-------------

You can implement your own logger by writing a class that inherits from
`LightningLoggerBase`. Use the `rank_zero_only` decorator to make sure that
only the first process in DDP training logs data.

.. code-block:: python

    from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

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


If you write a logger than may be useful to others, please send
a pull request to add it to Lighting!

Using loggers
-------------

You can call the logger anywhere from your LightningModule by doing:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_method_summary_writer_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.add_histogram(...)

Display metrics in progress bar
-------------------------------

.. code-block:: python

    # DEFAULT
    trainer = Trainer(show_progress_bar=True)

Log metric row every k batches
------------------------------

Every k batches lightning will make an entry in the metrics log

.. code-block:: python

    # DEFAULT (ie: save a .csv log file every 10 batches)
    trainer = Trainer(row_log_interval=10)

Log GPU memory
--------------

Logs GPU memory when metrics are logged.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(log_gpu_memory=None)

    # log only the min/max utilization
    trainer = Trainer(log_gpu_memory='min_max')

    # log all the GPU memory (if on DDP, logs only that node)
    trainer = Trainer(log_gpu_memory='all')

Process position
----------------

When running multiple models on the same machine we want to decide which progress bar to use.
 Lightning will stack progress bars according to this value.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(process_position=0)

    # if this is the second model on the node, show the second progress bar below
    trainer = Trainer(process_position=1)


Save a snapshot of all hyperparameters
--------------------------------------

Automatically log hyperparameters stored in the `hparams` attribute as an `argparse.Namespace`

.. code-block:: python

    class MyModel(pl.Lightning):
        def __init__(self, hparams):
            self.hparams = hparams

        ...

    args = parser.parse_args()
    model = MyModel(args)

    logger = TestTubeLogger(...)
    t = Trainer(logger=logger)
    trainer.fit(model)

Write logs file to csv every k batches
--------------------------------------

Every k batches, lightning will write the new logs to disk

.. code-block:: python

    # DEFAULT (ie: save a .csv log file every 100 batches)
    trainer = Trainer(log_save_interval=100)

"""

from os import environ
from .base import LightningLoggerBase, rank_zero_only

from .tensorboard import TensorBoardLogger

try:
    from .test_tube import TestTubeLogger
except ImportError:
    pass

try:
    from .mlflow import MLFlowLogger
except ImportError:
    pass

try:
    from .wandb import WandbLogger
except ImportError:
    pass
try:
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

    from .comet import CometLogger
except ImportError:
    del environ["COMET_DISABLE_AUTO_LOGGING"]

try:
    from .neptune import NeptuneLogger
except ImportError:
    pass
