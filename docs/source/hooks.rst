Model Hooks
===========

There are cases when you might want to do something different at different parts of the training/validation loop.
To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:

1. Fork `PyTorchLightning <https://github.com/PyTorchLightning/pytorch-lightning>`_.

2. Add the hook to :class:`pytorch_lightning.core.hooks.ModelHooks`.

3. Add it in the correct place in :mod:`pytorch_lightning.trainer` where it should be called.


Hooks lifecycle
---------------

Training set-up
^^^^^^^^^^^^^^^

- :meth:`~pytorch_lightning.core.lightning.LightningModule.init_ddp_connection`
- :meth:`~pytorch_lightning.trainer.optimizers.TrainerOptimizersMixin.init_optimizers`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_apex`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_ddp`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.test_dataloader`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.val_dataloader`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.summarize`
- :meth:`~pytorch_lightning.trainer.training_io.TrainerIOMixin.restore_weights`

Training loop
^^^^^^^^^^^^^

- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_epoch_start`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_batch_start`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step_end` (optional)
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.backward`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_after_backward`
- ``optimizer.step()``
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_batch_end`
- :meth:`~pytorch_lightning.core.lightning.LightningModule.training_epoch_end`
- :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_epoch_end`

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



.. automodule:: pytorch_lightning.core.hooks
    :noindex: