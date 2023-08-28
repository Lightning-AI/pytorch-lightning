:orphan:

.. _model_init:

########################
Efficient initialization
########################

Here are common use cases where you should use Lightning's initialization tricks to avoid major speed and memory bottlenecks when initializing your model.


----


**************
Half-precision
**************

Instantiating a ``nn.Module`` in PyTorch creates all parameters on CPU in float32 precision by default.
To speed up initialization, you can force PyTorch to create the model directly on the target device and with the desired precision without changing your model code.

.. code-block:: python

    trainer = Trainer(accelerator="cuda", precision="16-true")

    with trainer.init_module():
        # models created here will be on GPU and in float16
        model = MyLightningModule()

The larger the model, the more noticeable is the impact on

- **speed:** avoids redundant transfer of model parameters from CPU to device, avoids redundant casting from float32 to half precision
- **memory:** reduced peak memory usage since model parameters are never stored in float32


----


***********************************************
Loading checkpoints for inference or finetuning
***********************************************

When loading a model from a checkpoint, for example when fine-tuning, set ``empty_init=True`` to avoid expensive and redundant memory initialization:

.. code-block:: python

    with trainer.init_module(empty_init=True):
        # creation of the model is fast
        # and depending on the strategy allocates no memory, or uninitialized memory
        model = MyLightningModule.load_from_checkpoint("my/checkpoint/path.ckpt")

    trainer.fit(model)


.. warning::
    This is safe if you are loading a checkpoint that includes all parameters in the model.
    If you are loading a partial checkpoint (``strict=False``), you may end up with a subset of parameters that have uninitialized weights, unless you handle them accordingly.


----


********************************************
Model-parallel training (FSDP and DeepSpeed)
********************************************

When training sharded models with :ref:`FSDP <fully-sharded-training>` or :ref:`DeepSpeed <deepspeed_advanced>`, :meth:`~lightning.pytorch.trainer.trainer.Trainer.init_module` **should not be used**.
Instead, override the :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook:

.. code-block:: python

    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()
            # don't instantiate layers here
            # move the creation of layers to `configure_model`

        def configure_model(self):
            # create all your layers here
            self.layers = nn.Sequential(...)


Delaying the creation of large layers to the ``configure_model`` hook is necessary in most cases because otherwise initialization gets very slow (minutes) or (and that's more likely) you run out of CPU memory due to the size of the model.
