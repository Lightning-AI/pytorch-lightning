.. _test_set:

:orphan:

########################################
Validate and test a model (intermediate)
########################################

During and after training we need a way to evaluate our models to make sure they are not overfitting while training and
generalize well on unseen or real-world data. There are generally 2 stages of evaluation: validation and testing. To some
degree they serve the same purpose, to make sure models works on real data but they have some practical differences.

Validation is usually done during training, traditionally after each training epoch. It can be used for hyperparameter optimization or tracking model performance during training.
It's a part of the training process.

Testing is usually done once we are satisfied with the training and only with the best model selected from the validation metrics.

Let's see how these can be performed with Lightning.

*******
Testing
*******

Lightning allows the user to test their models with any compatible test dataloaders. This can be done before/after training
and is completely agnostic to :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit` call. The logic used here is defined under
:meth:`~lightning.pytorch.core.LightningModule.test_step`.

Testing is performed using the ``Trainer`` object's ``.test()`` method.

.. automethod:: lightning.pytorch.trainer.Trainer.test
    :noindex:


Test after Fit
==============

To run the test set after training completes, use this method.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # (1) load the best checkpoint automatically (lightning tracks this for you during .fit())
    trainer.test(ckpt_path="best")

    # (2) load the last available checkpoint (only works if `ModelCheckpoint(save_last=True)`)
    trainer.test(ckpt_path="last")

    # (3) test using a specific checkpoint
    trainer.test(ckpt_path="/path/to/my_checkpoint.ckpt")

    # (4) test with an explicit model (will use this model and not load a checkpoint)
    trainer.test(model)

.. warning::

    It is recommended to test with ``Trainer(devices=1)`` since distributed strategies such as DDP
    use :class:`~torch.utils.data.distributed.DistributedSampler` internally, which replicates some samples to
    make sure all devices have same batch size in case of uneven inputs. This is helpful to make sure
    benchmarking for research papers is done the right way.


Test Multiple Models
====================

You can run the test set on multiple models using the same trainer instance.

.. code-block:: python

    model1 = LitModel()
    model2 = GANModel()

    trainer = Trainer()
    trainer.test(model1)
    trainer.test(model2)


Test Pre-Trained Model
======================

To run the test set on a pre-trained model, use this method.

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path="/path/to/pytorch_checkpoint.ckpt",
        hparams_file="/path/to/experiment/version/hparams.yaml",
        map_location=None,
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this  case, the options you pass to trainer will be used when
running the test set (ie: 16-bit, dp, ddp, etc...)


Test with Additional DataLoaders
================================

You can still run inference on a test dataset even if the :meth:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader` method hasn't been
defined within your :doc:`lightning module <../common/lightning_module>` instance. This would be the case when your test data
is not available at the time your model was declared.

.. code-block:: python

    # setup your data loader
    test_dataloader = DataLoader(...)

    # test (pass in the loader)
    trainer.test(dataloaders=test_dataloader)

You can either pass in a single dataloader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases. Additionally,
you can also pass in an :doc:`datamodules <../data/datamodule>` that have overridden the
:ref:`datamodule_test_dataloader_label` method.

.. code-block:: python

    class MyDataModule(L.LightningDataModule):
        ...

        def test_dataloader(self):
            return DataLoader(...)


    # setup your datamodule
    dm = MyDataModule(...)

    # test (pass in datamodule)
    trainer.test(datamodule=dm)


Test with Multiple DataLoaders
==============================

When you need to evaluate your model on multiple test datasets simultaneously (e.g., different domains, conditions, or
evaluation scenarios), PyTorch Lightning supports multiple test dataloaders out of the box.

To use multiple test dataloaders, simply return a list of dataloaders from your ``test_dataloader()`` method:

.. code-block:: python

    class LitModel(L.LightningModule):
        def test_dataloader(self):
            return [
                DataLoader(clean_test_dataset, batch_size=32),
                DataLoader(noisy_test_dataset, batch_size=32),
                DataLoader(adversarial_test_dataset, batch_size=32),
            ]

When using multiple test dataloaders, your ``test_step`` method **must** include a ``dataloader_idx`` parameter:

.. code-block:: python

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Use dataloader_idx to handle different test scenarios
        return {'test_loss': loss}

Logging Metrics Per Dataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightning provides automatic support for logging metrics per dataloader:

.. code-block:: python

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        # Lightning automatically adds "/dataloader_idx_X" suffix
        self.log('test_loss', loss, add_dataloader_idx=True)
        self.log('test_acc', acc, add_dataloader_idx=True)

        return loss

This will create metrics like ``test_loss/dataloader_idx_0``, ``test_loss/dataloader_idx_1``, etc.

For more meaningful metric names, you can use custom naming where you need to make sure that individual names are
unique across dataloaders.

.. code-block:: python

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # Define meaningful names for each dataloader
        dataloader_names = {0: "clean", 1: "noisy", 2: "adversarial"}
        dataset_name = dataloader_names.get(dataloader_idx, f"dataset_{dataloader_idx}")

        # Log with custom names
        self.log(f'test_loss_{dataset_name}', loss, add_dataloader_idx=False)
        self.log(f'test_acc_{dataset_name}', acc, add_dataloader_idx=False)

Processing Entire Datasets Per Dataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform calculations on the entire test dataset for each dataloader (e.g., computing overall metrics, creating
visualizations), accumulate results during ``test_step`` and process them in ``on_test_epoch_end``:

.. code-block:: python

    class LitModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            # Store outputs per dataloader
            self.test_outputs = {}

        def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

            # Initialize and store results
            if dataloader_idx not in self.test_outputs:
                self.test_outputs[dataloader_idx] = {'predictions': [], 'targets': []}
            self.test_outputs[dataloader_idx]['predictions'].append(y_hat)
            self.test_outputs[dataloader_idx]['targets'].append(y)
            return loss

        def on_test_epoch_end(self):
            for dataloader_idx, outputs in self.test_outputs.items():
                # Concatenate all predictions and targets for this dataloader
                all_predictions = torch.cat(outputs['predictions'], dim=0)
                all_targets = torch.cat(outputs['targets'], dim=0)

                # Calculate metrics on the entire dataset, log and create visualizations
                overall_accuracy = (all_predictions.argmax(dim=1) == all_targets).float().mean()
                self.log(f'test_overall_acc_dataloader_{dataloader_idx}', overall_accuracy)
                self._save_results(all_predictions, all_targets, dataloader_idx)

            self.test_outputs.clear()

.. note::
    When using multiple test dataloaders, ``trainer.test()`` returns a list of results, one for each dataloader:

    .. code-block:: python

        results = trainer.test(model)
        print(f"Results from {len(results)} test dataloaders:")
        for i, result in enumerate(results):
            print(f"Dataloader {i}: {result}")

----------

**********
Validation
**********

Lightning allows the user to validate their models with any compatible ``val dataloaders``. This can be done before/after training.
The logic associated to the validation is defined within the :meth:`~lightning.pytorch.core.LightningModule.validation_step`.

Apart from this ``.validate`` has same API as ``.test``, but would rely respectively on :meth:`~lightning.pytorch.core.LightningModule.validation_step` and :meth:`~lightning.pytorch.core.LightningModule.test_step`.

.. note::
    ``.validate`` method uses the same validation logic being used under validation happening within
    :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit` call.

.. warning::

    When using ``trainer.validate()``, it is recommended to use ``Trainer(devices=1)`` since distributed strategies such as DDP
    uses :class:`~torch.utils.data.distributed.DistributedSampler` internally, which replicates some samples to
    make sure all devices have same batch size in case of uneven inputs. This is helpful to make sure
    benchmarking for research papers is done the right way.

.. automethod:: lightning.pytorch.trainer.Trainer.validate
    :noindex:
