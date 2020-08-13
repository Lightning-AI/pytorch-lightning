.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.base import Callback

.. role:: hidden
    :class: hidden-section

.. _callbacks:

Callbacks
=========

Lightning has a callback system to execute arbitrary code. Callbacks should capture NON-ESSENTIAL
logic that is NOT required for your :class:`~pytorch_lightning.core.LightningModule` to run.

An overall Lightning system should have:

1. Trainer for all engineering
2. LightningModule for all research code.
3. Callbacks for non-essential code.

|

Example:

.. testcode::

    class MyPrintingCallback(Callback):

        def on_init_start(self, trainer):
            print('Starting to init trainer!')

        def on_init_end(self, trainer):
            print('trainer is init now')

        def on_train_end(self, trainer, pl_module):
            print('do something when training ends')

    trainer = Trainer(callbacks=[MyPrintingCallback()])

.. testoutput::

    Starting to init trainer!
    trainer is init now

We successfully extended functionality without polluting our super clean
:class:`~pytorch_lightning.core.LightningModule` research code.

--------------

.. automodule:: pytorch_lightning.callbacks.base
   :noindex:
   :exclude-members:
        _del_model,
        _save_model,
        _abc_impl,
        check_monitor_top_k,

----------------

Built-in Callbacks
------------------
Lightning has a few built-in callbacks.

.. note::
    For a richer collection of callbacks, check out our
    `bolts library <https://pytorch-lightning-bolts.readthedocs.io/en/latest/callbacks.html>`_.

----------------

.. automodule:: pytorch_lightning.callbacks.early_stopping
   :noindex:
   :exclude-members:
        _del_model,
        _save_model,
        _abc_impl,
        check_monitor_top_k,

----------------

.. automodule:: pytorch_lightning.callbacks.gradient_accumulation_scheduler
   :noindex:
   :exclude-members:
        _del_model,
        _save_model,
        _abc_impl,
        check_monitor_top_k,

----------------

.. automodule:: pytorch_lightning.callbacks.lr_logger
    :noindex:
    :exclude-members:
        _extract_lr,
        _find_names

----------------

.. automodule:: pytorch_lightning.callbacks.model_checkpoint
   :noindex:
   :exclude-members:
        _del_model,
        _save_model,
        _abc_impl,
        check_monitor_top_k,

----------------

.. automodule:: pytorch_lightning.callbacks.progress
   :noindex:
   :exclude-members:

-----------

Examples
--------
You can do pretty much anything with callbacks.

- `Add a MLP to fine-tune self-supervised networks <https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_callbacks.html#sslonlineevaluator>`_.
- `Find how to modify an image input to trick the classification result <https://pytorch-lightning-bolts.readthedocs.io/en/latest/vision_callbacks.html#confused-logit>`_.
- `Interpolate the latent space of any variational model <https://pytorch-lightning-bolts.readthedocs.io/en/latest/variational_callbacks.html#latent-dim-interpolator>`_.
- `Log images to Tensorboard for any mode <https://pytorch-lightning-bolts.readthedocs.io/en/latest/vision_callbacks.html#tensorboard-image-generator>`_.

----------

Best Practices
--------------
The following are best practices when using/designing callbacks.

1. Callbacks should be isolated in their functionality.
2. Your callback should not rely on the behavior of other callbacks in order to work properly.
3. Do not manually call methods from the callback.
4. Directly calling methods (eg. `on_validation_end`) is strongly discouraged.
5. Whenever possible, your callbacks should not depend on the order in which they are executed.
