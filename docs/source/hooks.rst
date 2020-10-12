.. _hooks:

Model Hooks
===========

There are cases when you might want to do something different at different parts of the training/validation loop.
To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:

1. Fork `PyTorchLightning <https://github.com/PyTorchLightning/pytorch-lightning>`_.

2. Add the hook to :class:`pytorch_lightning.core.hooks.ModelHooks`.

3. Add it in the correct place in :mod:`pytorch_lightning.trainer` where it should be called.

----------------

Hooks lifecycle
---------------

Training set-up
^^^^^^^^^^^^^^^

- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.prepare_data`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.setup`
- :meth:`~pytorch_lightning.trainer.optimizers.TrainerOptimizersMixin.init_optimizers`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_apex`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.train_dataloader`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.test_dataloader`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.summarize`
- :meth:`~pytorch_lightning.trainer.training_io.TrainerIOMixin.restore_weights`

.. warning:: `prepare_data` is only called from global_rank=0. Don't assign state (self.something), use `setup` for that

----------

Training loop
^^^^^^^^^^^^^

- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_epoch_start`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step_end` (optional)
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.backward`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_after_backward`
- ``optimizer.step()``
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_train_batch_end`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_epoch_end`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_epoch_end`

----------

Validation loop
^^^^^^^^^^^^^^^

- ``model.zero_grad()``
- ``model.eval()``
- ``torch.set_grad_enabled(False)``
- :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step_end`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`
- ``model.train()``
- ``torch.set_grad_enabled(True)``
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_post_performance_check`

----------

Test loop
^^^^^^^^^

- ``model.zero_grad()``
- ``model.eval()``
- ``torch.set_grad_enabled(False)``
- :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step_end`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`
- ``model.train()``
- ``torch.set_grad_enabled(True)``
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_post_performance_check`

----------------

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
