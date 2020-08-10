Result
======
Lightning has two results objects `TrainResult` and `EvalResult`.

Use these to control:

- When to log (each step and/or epoch aggregate).
- Where to log (progress bar or a logger).
- How to sync across accelerators.

------------------

Training loop example
---------------------
Return a `TrainResult` from the Training loop.

.. code-block:: python

    def training_step(self, batch_subset, batch_idx):
        loss = ...
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

If you'd like to do something special with the outputs other than logging, implement `__epoch_end`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        result = pl.TrainResult(loss)
        result.some_prediction = some_prediction
        return result

    def training_epoch_end(self, training_step_output_result):
        all_train_predictions = training_step_output_result.some_prediction

        training_step_output_result.some_new_prediction = some_new_prediction
        return training_step_output_result

--------------------

Validation/Test loop example
-----------------------------
Return a `EvalResult` object from a validation/test loop

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        some_metric = ...
        result = pl.EvalResult(checkpoint_on=some_metric)
        result.log('some_metric', some_metric, prog_bar=True)
        return result

If you'd like to do something special with the outputs other than logging, implement `__epoch_end`.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        result = pl.EvalResult(checkpoint_on=some_metric)
        result.a_prediction = some_prediction
        return result

    def validation_epoch_end(self, validation_step_output_result):
        all_validation_step_predictions = validation_step_output_result.a_prediction
        # do something with the predictions from all validation_steps

        return validation_step_output_result


With the equivalent using the `EvalResult` syntax


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

Sync across devices
^^^^^^^^^^^^^^^^^^^
When training on multiple GPUs/CPUs/TPU cores, calculate the global mean of a logged metric as follows:

.. code-block:: python

    result.log('train_loss', loss, sync_dist=True)

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

Sync across devices
^^^^^^^^^^^^^^^^^^^
When training on multiple GPUs/CPUs/TPU cores, calculate the global mean of a logged metric as follows:

.. code-block:: python

    result.log('val_loss', loss, sync_dist=True)

EvalResult API
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.core.step_result.EvalResult
   :noindex:
