.. role:: hidden
    :class: hidden-section

.. _callbacks:

########
Callback
########

.. raw:: html

    <video width="100%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/callbacks.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/callbacks.mp4"></video>

|

A callback is a self-contained program that can be reused across projects.

Lightning has a callback system to execute them when needed. Callbacks should capture NON-ESSENTIAL
logic that is NOT required for your :doc:`lightning module <../common/lightning_module>` to run.

Here's the flow of how the callback hooks are executed:

.. raw:: html

    <video width="100%" max-width="400px" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_callbacks_mov.m4v"></video>

An overall Lightning system should have:

1. Trainer for all engineering
2. LightningModule for all research code.
3. Callbacks for non-essential code.

|

Example:

.. testcode::

    from pytorch_lightning.callbacks import Callback


    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is starting")

        def on_train_end(self, trainer, pl_module):
            print("Training is ending")


    trainer = Trainer(callbacks=[MyPrintingCallback()])

We successfully extended functionality without polluting our super clean
:doc:`lightning module <../common/lightning_module>` research code.

-----------

********
Examples
********
You can do pretty much anything with callbacks.

- `Add a MLP to fine-tune self-supervised networks <https://lightning-bolts.readthedocs.io/en/stable/deprecated/callbacks/self_supervised.html#sslonlineevaluator>`_.
- `Find how to modify an image input to trick the classification result <https://lightning-bolts.readthedocs.io/en/stable/deprecated/callbacks/vision.html#confused-logit>`_.
- `Interpolate the latent space of any variational model <https://lightning-bolts.readthedocs.io/en/stable/deprecated/callbacks/variational.html#latent-dim-interpolator>`_.
- `Log images to Tensorboard for any model <https://lightning-bolts.readthedocs.io/en/stable/deprecated/callbacks/vision.html#tensorboard-image-generator>`_.


--------------

******************
Built-in Callbacks
******************
Lightning has a few built-in callbacks.

.. note::
    For a richer collection of callbacks, check out our
    `bolts library <https://lightning-bolts.readthedocs.io/en/stable/index.html>`_.

.. currentmodule:: pytorch_lightning.callbacks

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    BackboneFinetuning
    BaseFinetuning
    BasePredictionWriter
    Callback
    DeviceStatsMonitor
    EarlyStopping
    GradientAccumulationScheduler
    LambdaCallback
    LearningRateMonitor
    ModelCheckpoint
    ModelPruning
    ModelSummary
    ProgressBarBase
    QuantizationAwareTraining
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    Timer
    TQDMProgressBar

----------

.. include:: callbacks_state.rst

----------


**************
Best Practices
**************
The following are best practices when using/designing callbacks.

1. Callbacks should be isolated in their functionality.
2. Your callback should not rely on the behavior of other callbacks in order to work properly.
3. Do not manually call methods from the callback.
4. Directly calling methods (eg. `on_validation_end`) is strongly discouraged.
5. Whenever possible, your callbacks should not depend on the order in which they are executed.


-----------

.. include:: entry_points.rst

-----------

.. _callback_hooks:

************
Callback API
************
Here is the full API of methods available in the Callback base class.

The :class:`~pytorch_lightning.callbacks.Callback` class is the base for all the callbacks in Lightning just like the :class:`~pytorch_lightning.core.module.LightningModule` is the base for all models.
It defines a public interface that each callback implementation must follow, the key ones are:

Properties
==========

state_key
^^^^^^^^^

.. autoattribute:: pytorch_lightning.callbacks.Callback.state_key
    :noindex:


Hooks
=====

on_configure_sharded_model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_configure_sharded_model
    :noindex:

setup
^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.setup
    :noindex:

teardown
^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.teardown
    :noindex:

on_init_start
^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_init_start
    :noindex:

on_init_end
^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_init_end
    :noindex:

on_fit_start
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_fit_start
    :noindex:

on_fit_end
^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_fit_end
    :noindex:

on_sanity_check_start
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_sanity_check_start
    :noindex:

on_sanity_check_end
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_sanity_check_end
    :noindex:

on_train_batch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_batch_start
    :noindex:

on_train_batch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_batch_end
    :noindex:

on_train_epoch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_epoch_start
    :noindex:

on_train_epoch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_epoch_end
    :noindex:

on_validation_epoch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_epoch_end
    :noindex:

on_test_epoch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_epoch_start
    :noindex:

on_test_epoch_end
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_epoch_end
    :noindex:

on_predict_epoch_start
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_epoch_start
    :noindex:

on_predict_epoch_end
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_epoch_end
    :noindex:

.. automethod:: pytorch_lightning.callbacks.Callback.on_epoch_end
    :noindex:

on_validation_batch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_batch_start
    :noindex:

on_validation_batch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_batch_end
    :noindex:

on_test_batch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_batch_start
    :noindex:

on_test_batch_end
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_batch_end
    :noindex:

on_predict_batch_start
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_batch_start
    :noindex:

on_predict_batch_end
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_batch_end
    :noindex:

on_train_start
^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_start
    :noindex:

on_train_end
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_train_end
    :noindex:

on_validation_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_start
    :noindex:

on_validation_end
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_validation_end
    :noindex:

on_test_start
^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_start
    :noindex:

on_test_end
^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_test_end
    :noindex:

on_predict_start
^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_start
    :noindex:

on_predict_end
^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_predict_end
    :noindex:

on_keyboard_interrupt
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_keyboard_interrupt
    :noindex:

on_exception
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_exception
    :noindex:

state_dict
^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.state_dict
    :noindex:

on_save_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_save_checkpoint
    :noindex:

load_state_dict
^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.load_state_dict
    :noindex:

on_load_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_load_checkpoint
    :noindex:

on_before_backward
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_before_backward
    :noindex:

on_after_backward
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_after_backward
    :noindex:

on_before_optimizer_step
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_before_optimizer_step
    :noindex:

on_before_zero_grad
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.callbacks.Callback.on_before_zero_grad
    :noindex:
