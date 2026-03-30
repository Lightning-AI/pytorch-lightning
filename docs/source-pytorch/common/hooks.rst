##########################
Hooks in PyTorch Lightning
##########################

Hooks in Pytorch Lightning allow you to customize the training, validation, and testing logic of your models. They
provide a way to insert custom behavior at specific points during the training process without modifying the core
training loop. There are several categories of hooks available in PyTorch Lightning:

1. **Setup/Teardown Hooks**: Called at the beginning and end of training phases
2. **Training Hooks**: Called during the training loop
3. **Validation Hooks**: Called during validation
4. **Test Hooks**: Called during testing
5. **Prediction Hooks**: Called during prediction
6. **Optimizer Hooks**: Called around optimizer operations
7. **Checkpoint Hooks**: Called during checkpoint save/load operations
8. **Exception Hooks**: Called when exceptions occur

Nearly all hooks can be implemented in three places within your code:

- **LightningModule**: The main module where you define your model and training logic.
- **Callbacks**: Custom classes that can be passed to the Trainer to handle specific events.
- **Strategy**: Custom strategies for distributed training.

Importantly, because logic can be place in the same hook but in different places the call order of hooks is in
important to understand. The following order is always used:

1. Callbacks, called in the order they are passed to the Trainer.
2. ``LightningModule``
3. Strategy

.. testcode::

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.demos import BoringModel

    class MyModel(BoringModel):
        def on_train_start(self):
            print("Model: Training is starting!")

    class MyCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Callback: Training is starting!")

    model = MyModel()
    callback = MyCallback()
    trainer = Trainer(callbacks=[callback], logger=False, max_epochs=1)
    trainer.fit(model)

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    ...
    Callback: Training is starting!
    Model: Training is starting!
    ...


.. note::
   There are a few exceptions to this pattern:

   - **on_train_epoch_end**: Non-monitoring callbacks are called first, then ``LightningModule``, then monitoring callbacks
   - **Optimizer hooks** (on_before_backward, on_after_backward, on_before_optimizer_step): Only callbacks and ``LightningModule`` are called
   - Some internal hooks may only call ``LightningModule`` or Strategy

************************
Training Loop Hook Order
************************

The following diagram shows the execution order of hooks during a typical training loop e.g. calling `trainer.fit()`,
with the source of each hook indicated:

.. code-block:: text

    Training Process Flow:

    trainer.fit()
    │
    ├── setup(stage="fit")
    │   ├── [LightningDataModule]
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   ├── [LightningModule.configure_shared_model()]
    │   ├── [LightningModule.configure_model()]
    │   ├── Strategy.restore_checkpoint_before_setup
    │   │   ├── [LightningModule.on_load_checkpoint()]
    │   │   ├── [LightningModule.load_state_dict()]
    │   │   ├── [LightningDataModule.load_state_dict()]
    │   │   ├── [Callbacks.on_load_checkpoint()]
    │   │   └── [Callbacks.load_state_dict()]
    │   └── [Strategy]
    │
    ├── on_fit_start()
    │   ├── [Callbacks]
    │   └── [LightningModule]
    │
    ├── Strategy.restore_checkpoint_after_setup
    │   ├── [LightningModule.on_load_checkpoint()]
    │   ├── [LightningModule.load_state_dict()]
    │   ├── [LightningDataModule.load_state_dict()]
    │   ├── [Callbacks.on_load_checkpoint()]
    │   └── [Callbacks.load_state_dict()]
    │
    ├── on_sanity_check_start()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │   ├── on_validation_start()
    │   │   ├── [Callbacks]
    │   │   ├── [LightningModule]
    │   │   └── [Strategy]
    │   ├── on_validation_epoch_start()
    │   │   ├── [Callbacks]
    │   │   ├── [LightningModule]
    │   │   └── [Strategy]
    │   │   ├── [for each validation batch]
    │   │   │   ├── on_validation_batch_start()
    │   │   │   │   ├── [Callbacks]
    │   │   │   │   ├── [LightningModule]
    │   │   │   │   └── [Strategy]
    │   │   │   └── on_validation_batch_end()
    │   │   │       ├── [Callbacks]
    │   │   │       ├── [LightningModule]
    │   │   │       └── [Strategy]
    │   │   └── [end validation batches]
    │   ├── on_validation_epoch_end()
    │   │   ├── [Callbacks]
    │   │   ├── [LightningModule]
    │   │   └── [Strategy]
    │   └── on_validation_end()
    │       ├── [Callbacks]
    │       ├── [LightningModule]
    │       └── [Strategy]
    ├── on_sanity_check_end()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │
    ├── on_train_start()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │
    ├── [Training Epochs Loop]
    │   │
    │   ├── on_train_epoch_start()
    │   │   ├── [Callbacks]
    │   │   └── [LightningModule]
    │   │
    │   ├── [Training Batches Loop]
    │   │   │
    │   │   ├── on_train_batch_start()
    │   │   │   ├── [Callbacks]
    │   │   │   ├── [LightningModule]
    │   │   │   └── [Strategy]
    │   │   │
    │   │   ├── [Forward Pass - training_step()]
    │   │   │   └── [Strategy only]
    │   │   │
    │   │   ├── on_before_zero_grad()
    │   │   │   ├── [Callbacks]
    │   │   │   └── [LightningModule]
    │   │   │
    │   │   ├── optimizer_zero_grad()
    │   │   │   └── [LightningModule only - optimizer_zero_grad()]
    │   │   │
    │   │   ├── [Backward Pass - Strategy.backward()]
    │   │   │   ├── on_before_backward()
    │   │   │   │   ├── [Callbacks]
    │   │   │   │   └── [LightningModule]
    │   │   │   ├── LightningModule.backward()
    │   │   │   └── on_after_backward()
    │   │   │       ├── [Callbacks]
    │   │   │       └── [LightningModule]
    │   │   │
    │   │   ├── on_before_optimizer_step()
    │   │   │   ├── [Callbacks]
    │   │   │   └── [LightningModule]
    │   │   │
    │   │   ├── [Optimizer Step]
    │   │   │   └── [LightningModule only - optimizer_step()]
    │   │   │
    │   │   └── on_train_batch_end()
    │   │       ├── [Callbacks]
    │   │       └── [LightningModule]
    │   │
    │   │   [Optional: Validation during training]
    │   │   ├── on_validation_start()
    │   │   │   ├── [Callbacks]
    │   │   │   ├── [LightningModule]
    │   │   │   └── [Strategy]
    │   │   ├── on_validation_epoch_start()
    │   │   │   ├── [Callbacks]
    │   │   │   ├── [LightningModule]
    │   │   │   └── [Strategy]
    │   │   │   ├── [for each validation batch]
    │   │   │   │   ├── on_validation_batch_start()
    │   │   │   │   │   ├── [Callbacks]
    │   │   │   │   │   ├── [LightningModule]
    │   │   │   │   │   └── [Strategy]
    │   │   │   │   └── on_validation_batch_end()
    │   │   │   │       ├── [Callbacks]
    │   │   │   │       ├── [LightningModule]
    │   │   │   │       └── [Strategy]
    │   │   │   └── [end validation batches]
    │   │   ├── on_validation_epoch_end()
    │   │   │   ├── [Callbacks]
    │   │   │   ├── [LightningModule]
    │   │   │   └── [Strategy]
    │   │   └── on_validation_end()
    │   │       ├── [Callbacks]
    │   │       ├── [LightningModule]
    │   │       └── [Strategy]
    │   │
    │   └── on_train_epoch_end() **SPECIAL CASE**
    │       ├── [Callbacks - Non-monitoring only]
    │       ├── [LightningModule]
    │       └── [Callbacks - Monitoring only]
    │
    ├── [End Training Epochs]
    │
    ├── on_train_end()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │
    └── teardown(stage="fit")
        ├── [Strategy]
        ├── on_fit_end()
        │   ├── [Callbacks]
        │   └── [LightningModule]
        ├── [LightningDataModule]
        ├── [Callbacks]
        └── [LightningModule]

***********************
Testing Loop Hook Order
***********************

When running tests with ``trainer.test()``:

.. code-block:: text

    trainer.test()
    │
    ├── setup(stage="test")
    │   └── [Callbacks only]
    ├── on_test_start()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │
    ├── [Test Epochs Loop]
    │   │
    │   ├── on_test_epoch_start()
    │   │   ├── [Callbacks]
    │   │   ├── [LightningModule]
    │   │   └── [Strategy]
    │   │
    │   ├── [Test Batches Loop]
    │   │   │
    │   │   ├── on_test_batch_start()
    │   │   │   ├── [Callbacks]
    │   │   │   ├── [LightningModule]
    │   │   │   └── [Strategy]
    │   │   │
    │   │   └── on_test_batch_end()
    │   │       ├── [Callbacks]
    │   │       ├── [LightningModule]
    │   │       └── [Strategy]
    │   │
    │   └── on_test_epoch_end()
    │       ├── [Callbacks]
    │       ├── [LightningModule]
    │       └── [Strategy]
    │
    ├── on_test_end()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    └── teardown(stage="test")
        └── [Callbacks only]

**************************
Prediction Loop Hook Order
**************************

When running predictions with ``trainer.predict()``:

.. code-block:: text

    trainer.predict()
    │
    ├── setup(stage="predict")
    │   └── [Callbacks only]
    ├── on_predict_start()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    │
    ├── [Prediction Epochs Loop]
    │   │
    │   ├── on_predict_epoch_start()
    │   │   ├── [Callbacks]
    │   │   └── [LightningModule]
    │   │
    │   ├── [Prediction Batches Loop]
    │   │   │
    │   │   ├── on_predict_batch_start()
    │   │   │   ├── [Callbacks]
    │   │   │   └── [LightningModule]
    │   │   │
    │   │   └── on_predict_batch_end()
    │   │       ├── [Callbacks]
    │   │       └── [LightningModule]
    │   │
    │   └── on_predict_epoch_end()
    │       ├── [Callbacks]
    │       └── [LightningModule]
    │
    ├── on_predict_end()
    │   ├── [Callbacks]
    │   ├── [LightningModule]
    │   └── [Strategy]
    └── teardown(stage="predict")
        └── [Callbacks only]
