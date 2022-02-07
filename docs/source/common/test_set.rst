.. _test_set:

##########
Evaluation
##########

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
and is completely agnostic to :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` call. The logic used here is defined under
:meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`.

Testing is performed using the ``Trainer`` object's ``.test()`` method.

.. automethod:: pytorch_lightning.trainer.Trainer.test
    :noindex:


Test after Fit
==============

To run the test set after training completes, use this method.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # (1) load the best checkpoint automatically (lightning tracks this for you)
    trainer.test(ckpt_path="best")

    # (2) test using a specific checkpoint
    trainer.test(ckpt_path="/path/to/my_checkpoint.ckpt")

    # (3) test with an explicit model (will use this model and not load a checkpoint)
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
        hparams_file="/path/to/test_tube/experiment/version/hparams.yaml",
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

You can still run inference on a test dataset even if the :meth:`~pytorch_lightning.core.hooks.DataHooks.test_dataloader` method hasn't been
defined within your :doc:`lightning module <../common/lightning_module>` instance. This would be the case when your test data
is not available at the time your model was declared.

.. code-block:: python

    # setup your data loader
    test_dataloader = DataLoader(...)

    # test (pass in the loader)
    trainer.test(dataloaders=test_dataloader)

You can either pass in a single dataloader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases. Additionally,
you can also pass in an :doc:`datamodules <../extensions/datamodules>` that have overridden the
:ref:`datamodule_test_dataloader_label` method.

.. code-block:: python

    class MyDataModule(pl.LightningDataModule):
        ...

        def test_dataloader(self):
            return DataLoader(...)


    # setup your datamodule
    dm = MyDataModule(...)

    # test (pass in datamodule)
    trainer.test(datamodule=dm)

----------

**********
Validation
**********

Lightning allows the user to validate their models with any compatible ``val dataloaders``. This can be done before/after training.
The logic associated to the validation is defined within the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`.

Apart from this ``.validate`` has same API as ``.test``, but would rely respectively on :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` and :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`.

.. note::
    ``.validate`` method uses the same validation logic being used under validation happening within
    :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` call.

.. warning::

    When using ``trainer.validate()``, it is recommended to use ``Trainer(devices=1)`` since distributed strategies such as DDP
    uses :class:`~torch.utils.data.distributed.DistributedSampler` internally, which replicates some samples to
    make sure all devices have same batch size in case of uneven inputs. This is helpful to make sure
    benchmarking for research papers is done the right way.

.. automethod:: pytorch_lightning.trainer.Trainer.validate
    :noindex:
