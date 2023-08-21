:orphan:

.. _model_init:

########################
Efficient initialization
########################

Instantiating a ``nn.Module`` in PyTorch creates all parameters on CPU in float32 precision by default.
To speed up initialization, you can force PyTorch to create the model directly on the target device and with the desired precision without changing your model code.

.. code-block:: python

    fabric = Trainer(accelerator="cuda", precision="16-true")

    with trainer.init_module():
        # models created here will be on GPU and in float16
        model = MyModel()

    trainer.fit(model)

This eliminates the waiting time to transfer the model parameters from the CPU to the device.

When loading a model from a checkpoint, for example when fine-tuning, set `empty_init=True` to avoid expensive
and redundant memory initialization:

.. code-block:: python

    with trainer.init_module(empty_init=True):
        # creation of the model is very fast
        model = MyModel.load_from_checkpoint("my/checkpoint/path.ckpt")

    trainer.fit(model)

For strategies that handle large sharded models (:ref:`FSDP <fully-sharded-training>`, :ref:`DeepSpeed <deepspeed_advanced>`), the :meth:`~lightning.pytorch.trainer.trainer.Trainer.init_module`
should not be used, instead override the :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook:

.. code-block:: python

    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()
            # don't instantiate layers here
            # move the creation of layers to `configure_model`

        def configure_model(self):
            # create all your layers here
            self.layers = nn.Sequential(...)

This makes it possible to work with models that are larger than the memory of a single device.
