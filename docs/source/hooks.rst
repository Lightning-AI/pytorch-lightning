.. _hooks:

Callback hooks
==============

There are cases when you might want to do something different at different parts of the training/validation loop.
To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:

1. Fork `PyTorchLightning <https://github.com/PyTorchLightning/pytorch-lightning>`_.

2. Add the hook to :class:`pytorch_lightning.core.hooks.ModelHooks`.

3. Add it in the correct place in :mod:`pytorch_lightning.trainer` where it should be called.

---------

General hooks
-------------

on_after_backward
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_after_backward
    :noindex:

on_before_zero_grad
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad
    :noindex:

on_epoch_start
^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_epoch_start
    :noindex:

on_epoch_end
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_epoch_end
    :noindex:

on_fit_start
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_fit_start
    :noindex:

on_fit_end
^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_fit_end
    :noindex:

on_save_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint
    :noindex:

on_load_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint
    :noindex:

on_pretrain_routine_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_pretrain_routine_start
    :noindex:

on_pretrain_routine_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_pretrain_routine_end
    :noindex:

on_test_batch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_batch_start
    :noindex:

on_test_batch_end
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_batch_end
    :noindex:

on_test_epoch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_epoch_start
    :noindex:

on_test_epoch_end
^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_epoch_end
    :noindex:

on_test_model_train
^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_model_train
    :noindex:

on_test_model_eval
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_model_eval
    :noindex:

on_train_batch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start
    :noindex:

on_train_batch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_batch_end
    :noindex:

on_train_start
^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_start
    :noindex:

on_train_end
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_end
    :noindex:

on_train_epoch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_epoch_start
    :noindex:

on_train_epoch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_epoch_end
    :noindex:

on_validation_batch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_batch_start
    :noindex:

on_validation_batch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_batch_end
    :noindex:

on_validation_epoch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_epoch_end
    :noindex:

on_validation_model_eval
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval
    :noindex:

on_validation_model_train
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_model_train
    :noindex:

---------

Data hooks
----------

setup
^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.setup
    :noindex:

teardown
^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.teardown
    :noindex:

prepare_data
^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.DataHooks.prepare_data
    :noindex:

test_dataloader
^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.DataHooks.test_dataloader
    :noindex:

train_dataloader
^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.DataHooks.train_dataloader
    :noindex:

transfer_batch_to_device
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.DataHooks.transfer_batch_to_device
    :noindex:

val_dataloader
^^^^^^^^^^^^^^

.. automethod:: pytorch_lightning.core.hooks.DataHooks.val_dataloader
    :noindex:
