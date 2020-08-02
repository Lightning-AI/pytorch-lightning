Result
======
Lightning has two results objects `TrainResult` and `EvalResult` which can remove the need for
a `_step_end` or `_epoch_end` loop. If these loops are only used to aggregate logging statistics,
then replace those loops with the respective result object.

However, if you need fine-grain control to do more than logging or a complex aggregation, then keep
the loops as they are and do not use the `EvalResult` or `TrainResult` objects.

.. note:: These objects are optional and should only be used if you don't need full control of the loops.

Training loop example
---------------------
We can simplify the following multi-method training loop:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'epoch_loss': epoch_loss},
            'progress_bar': {'epoch_loss': epoch_loss}
        }

using the equivalent syntax via the `TrainResult` object:

.. code-block:: python

    def training_step(self, batch_subset, batch_idx):
        loss = ...
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

Validation loop example
-----------------------
We can replace the following validation/test loop:

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        return {'some_metric': some_metric}

    def validation_epoch_end(self, validation_step_outputs):
        some_metric_mean = torch.stack([x['some_metric'] for x in validation_step_outputs]).mean()
        return {
            'log': {'some_metric_mean': some_metric_mean},
            'progress_bar': {'some_metric_mean': some_metric_mean}
        }

With the equivalent using the `EvalResult` syntax

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        some_metric = ...
        result = pl.EvalResult(checkpoint_on=some_metric)
        result.log('some_metric', some_metric, prog_bar=True)
        return result

------------------

TrainResult
-----------
The `TrainResult` basic usage is this:

minimize
^^^^^^^^
.. code-block:: python

    def training_step(...):
        return TrainResult(some_metric)


checkpoint/early_stop
^^^^^^^^^^^^^^^^^^^^^
If you are only using a training loop (no val), you can also specify what to monitor for
checkpointing or early stopping:

.. code-block:: python

    def training_step(...):
        return TrainResult(some_metric, checkpoint_on=metric_a, early_stop_on=metric_b)


In the manual loop, checkpoint and early stop is based only on the loss returned. With the `TrainResult` you
can change it every batch if you want, or even monitor different metrics for each purpose.

.. code-block:: python

    # early stop + checkpoint can only use the `loss` when done manually via dictionaries
    def training_step(...):
        return loss
    def training_step(...):
        return {'loss' loss}

logging
^^^^^^^
The main benefit of the `TrainResult` is automatic logging at whatever level you want.

.. code-block:: python

    result = TrainResult(loss)
    result.log('train_loss', loss)

    # equivalent
    result.log('train_loss', loss, on_step=True, on_epoch=False, logger=True, prog_bar=False, reduce_fx=torch.mean)

By default, any log calls will log only that step's metrics to the logger. To change when and where to log
update the defaults as needed.

Change where to log:

.. code-block:: python

    # to logger only (default)
    result.log('train_loss', loss)

    # logger + progress bar
    result.log('train_loss', loss, prog_bar=True)

    # progress bar only
    result.log('train_loss', loss, prog_bar=True, logger=False)

Sometimes you may also want to get epoch level statistics:

.. code-block:: python

    # loss at this step
    result.log('train_loss', loss)

    # loss for the epoch
    result.log('train_loss', loss, on_step=False, on_epoch=True)

    # loss for the epoch AND step
    # the logger will show 2 charts: step_train_loss, epoch_train_loss
    result.log('train_loss', loss, on_epoch=True)

Finally, you can use your own reduction function instead:

.. code-block:: python

    # the total sum for all batches of an epoch
    result.log('train_loss', loss, on_epoch=True, reduce_fx=torch.sum)

    def my_reduce_fx(all_train_loss):
        # reduce somehow
        return result

    result.log('train_loss', loss, on_epoch=True, reduce_fx=my_reduce_fx)

.. note:: Use this ONLY in the case where your loop is simple and simply logs.

Finally, you may need more esoteric logging such as something specific to your logger like images:


.. code-block:: python

    def training_step(...):
        result = TrainResult(some_metric)
        result.log('train_loss', loss)

        # also log images (if tensorboard for example)
        self.logger.experiment.log_figure(...)

TrainResult API
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.core.step_result.TrainResult
   :noindex:

------------------

EvalResult
----------
The `EvalResult` object has the same usage as the `TrainResult` object.

.. code-block:: python

    def validation_step(...):
        return EvalResult()

    def test_step(...):
        return EvalResult()

However, there are some differences:

Eval minimize
^^^^^^^^^^^^^
- There is no `minimize` argument (since we don't learn during validation)

Eval checkpoint/early_stopping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If defined in both the `TrainResult` and the `EvalResult` the one in the `EvalResult` will take precedence.

.. code-block:: python

    def training_step(...):
        return TrainResult(loss, checkpoint_on=metric, early_stop_on=metric)

    # metric_a and metric_b will be used for the callbacks and NOT metric
    def validation_step(...):
        return EvalResult(checkpoint_on=metric_a, early_stop_on=metric_b)

Eval logging
^^^^^^^^^^^^
Logging has the same behavior as `TrainResult` but the logging defaults are different:

.. code-block:: python

    # TrainResult logs by default at each step only
    TrainResult().log('val', val, on_step=True, on_epoch=False, logger=True, prog_bar=False, reduce_fx=torch.mean)

    # EvalResult logs by default at the end of an epoch only
    EvalResult().log('val', val, on_step=False, on_epoch=True, logger=True, prog_bar=False, reduce_fx=torch.mean)

Val/Test loop
^^^^^^^^^^^^^
Eval result can be used in both `test_step` and `validation_step`.

EvalResult API
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.core.step_result.EvalResult
   :noindex:
