.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning import loggers as pl_loggers

.. role:: hidden
    :class: hidden-section

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

Note::

    All loggers log by default to ``os.getcwd()``. To change the path without creating a logger set
    ``Trainer(default_root_dir='/your/path/to/save/checkpoints')``

----------

Logging from a LightningModule
------------------------------
Use the Result objects to log from any lightning module.

Training loop logging
^^^^^^^^^^^^^^^^^^^^^
To log in the training loop use the :class:`TrainResult`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

The `Result` object is simply a dictionary that gives you added methods like `log` and `write`
and automatically detaches tensors (except for the minimize value).

.. code-block:: python

    result = pl.TrainResult(minimize=loss)
    result.log('train_loss', loss)
    print(result)

    {'train_loss': tensor([0.2262])}

The `TrainResult` can log at two places in the training, on each step (`TrainResult(on_step=True)`) and
the aggregate at the end of the epoch (`TrainResult(on_epoch=True)`).

.. code-block:: python

    for epoch in epochs:
        epoch_outs = []
        for batch in train_dataloader():
            # ......
            out = training_step(batch)
            # < ----------- log (on_step=True)
            epoch_outs.append(out)

        # < -------------- log (on_epoch=True)
        auto_reduce_log(epoch_outs)

Validation loop logging
^^^^^^^^^^^^^^^^^^^^^^^
To log in the training loop use the :class:`EvalResult`.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        loss = ...

        result = pl.EvalResult()
        result.log('val_loss', loss)
        return result

The `EvalResult` object is simply a dictionary that gives you added methods like `log` and `write`
and automatically detaches tensors (except for the minimize value).

.. code-block:: python

    result = pl.EvalResult()
    result.log('val_loss', loss)
    print(result)

    {'val_loss': tensor([0.2262])}

The `EvalResult` can log at two places in the validation loop, on each step (`EvalResult(on_step=True)`) and
the aggregate at the end of the epoch (`EvalResult(on_epoch=True)`).

.. code-block:: python

    def run_val_loop():
        epoch_outs = []
        for batch in val_dataloader():
            out = validation_step(batch)
            # < ----------- log (on_step=True)
            epoch_outs.append(out)

        # < -------------- log (on_epoch=True)
        auto_reduce_log(epoch_outs)

Test loop logging
^^^^^^^^^^^^^^^^^
See the previous section.

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

This also applies to Callbacks


----------

Logging from a Callback
-----------------------
To log from a callback, access the logger object directly

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
            pass

        @rank_zero_only
        def finalize(self, status):
            # Optional. Any code that needs to be run after training
            # finishes goes here
            pass

If you write a logger that may be useful to others, please send
a pull request to add it to Lighting!

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
