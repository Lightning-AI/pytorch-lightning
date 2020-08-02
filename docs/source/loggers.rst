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

Logging from a LightningModule
------------------------------
There are two ways to get a LightningModule to log.

- The first is manual logging where you control the aggregation of metrics.
- The second way is where you defer the aggregation to the Trainer.

Training loop logging
^^^^^^^^^^^^^^^^^^^^^

Train Manual logging
*********************
Use this approach when you want fine-control over your logging.

For the training loop, return the 'log' or 'progress_bar' keywords.

- `log` sends to the `self.logger` object (tensorboard, etc)
- `progress_bar` sends to the progress bar display

.. code-block:: python

    def training_step(self, batch, batch_idx):
        return {
            'loss': loss,
            'log': {'train_loss': loss, 'batch_accuracy': accuracy}
        }

If using DataParallel or DDP2, you may want to return metrics for the full batch and not just the batch subset

.. code-block:: python

    def training_step(self, batch_subset, batch_idx):
        return {'train_step_preds': pred, 'train_step_target': target}

    def training_step_end(self, all_batch_outputs):
        batch_acc = 0
        batch_preds = torch.stack([x['train_step_preds'] for x in all_batch_outputs])
        batch_targets = torch.stack([x['train_step_target'] for x in all_batch_outputs])

        batch_loss = F.cross_entropy(batch_preds, batch_targets)
        batch_acc = metrics.functional.accuracy(batch_preds, batch_targets)

        return {
            'loss': batch_loss,
            'log': {'train_loss': batch_loss, 'batch_accuracy': batch_acc},
            'progress_bar': {'train_loss': batch_loss, 'batch_accuracy': batch_acc}
        }

Or if you need epoch level metrics, you can also implement the `epoch_end` method

.. code-block:: python

    def training_step(self, batch, batch_idx):
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()

.. note:: After the `training_step` output is processed, we call detach on the loss so that memory remains low. The
    inputs to `training_epoch_end` have all detached losses

Train automatic logging
***********************
If you do not need to do anything special except reduce metrics, you can use the `TrainResult` object to do
the aggregation for you. This means you won't need the `step_end` or `epoch_end` method.

`training_step` only: These are equivalent,

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...
        return {'loss': loss, 'log': {'train_loss': loss}}

    # ------------
    # equivalent
    # ------------
    def training_step(self, batch, batch_idx):
        loss = ...

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

`training_step` + `training_step_end`: These are also equivalent,

.. code-block:: python

    def training_step(self, batch_subset, batch_idx):
        return {'train_step_preds': pred, 'train_step_target': target}

    def training_step_end(self, all_batch_outputs):
        batch_acc = 0
        batch_preds = torch.stack([x['train_step_preds'] for x in all_batch_outputs])
        batch_targets = torch.stack([x['train_step_target'] for x in all_batch_outputs])

        batch_loss = F.cross_entropy(batch_preds, batch_targets)
        batch_acc = metrics.functional.accuracy(batch_preds, batch_targets)

        return {
            'loss': batch_loss,
            'log': {'train_loss': batch_loss, 'batch_accuracy': batch_acc},
            'progress_bar': {'train_loss': batch_loss, 'batch_accuracy': batch_acc}
        }

    # ------------
    # equivalent
    # ------------
    def training_step(self, batch_subset, batch_idx):
        loss = ...
        batch_subset_acc = ...

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log('batch_acc', batch_subset_acc, prog_bar=True)
        return result

`training_step` + `training_epoch_end`: These are also equivalent,

.. code-block:: python

    def training_step(self, batch, batch_idx):
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'epoch_loss': epoch_loss},
            'progress_bar': {'epoch_loss': epoch_loss}
        }

    # ------------
    # equivalent
    # ------------
    def training_step(self, batch, batch_idx):
        loss = ...
        batch_subset_acc = ...

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return result

Validation loop logging
^^^^^^^^^^^^^^^^^^^^^^^

Val manual logging
******************
In manual logging, only the output of `validation_epoch_end` is used for logging. The reason is that during
validation, the model is not learning, so each batch is treated independently and thus epoch metrics
don't really make sense unless you want a histogram.

To log, you need two methods, `validation_step` where you compute the metrics per batch, and `validation_epoch_end`
where you aggregate them.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        return {'some_metric': some_metric}

    def validation_epoch_end(self, validation_step_outputs):
        some_metric_mean = torch.stack([x['some_metric'] for x in validation_step_outputs]).mean()
        return {
            'log': {'some_metric_mean': some_metric_mean},
            'progress_bar': {'some_metric_mean': some_metric_mean}
        }

in `dp` or `ddp2` mode (DataParallel), feel free to also use the `validation_step_end` method to aggregate for the
batch as was shown in `training_step_end`.

Val automatic logging
*********************
The above is a lot of work if you're not doing anything special in your validation loop, other than just logging.
In that case, use the `EvalResult` object:

The `EvalResult` removes the need for the `step_end` or `epoch_end` method unless you really
need it (as described above)

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        return {'some_metric': some_metric}

    def validation_epoch_end(self, validation_step_outputs):
        some_metric_mean = torch.stack([x['some_metric'] for x in validation_step_outputs]).mean()
        return {
            'log': {'some_metric_mean': some_metric_mean},
            'progress_bar': {'some_metric_mean': some_metric_mean}
        }

    # ------------
    # equivalent
    # ------------
    def validation_step(self, batch, batch_idx):
        some_metric = ...
        result = pl.EvalResult(checkpoint_on=some_metric)
        result.log('some_metric', some_metric, prog_bar=True)
        return result


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