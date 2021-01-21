.. _test_set:

Test set
========
Lightning forces the user to run the test set separately to make sure it isn't evaluated by mistake.
Testing is performed using the ``trainer`` object's ``.test()`` method.

.. automethod:: pytorch_lightning.trainer.Trainer.test
    :noindex:

----------

Test after fit
--------------
To run the test set after training completes, use this method.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # (1) load the best checkpoint automatically (lightning tracks this for you)
    trainer.test()

    # (2) don't load a checkpoint, instead use the model with the latest weights
    trainer.test(ckpt_path=None)

    # (3) test using a specific checkpoint
    trainer.test(ckpt_path='/path/to/my_checkpoint.ckpt')

    # (4) test with an explicit model (will use this model and not load a checkpoint)
    trainer.test(model)

----------

Test multiple models
--------------------
You can run the test set on multiple models using the same trainer instance.

.. code-block:: python

    model1 = LitModel()
    model2 = GANModel()

    trainer = Trainer()
    trainer.test(model1)
    trainer.test(model2)

----------

Test pre-trained model
----------------------
To run the test set on a pre-trained model, use this method.

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path='/path/to/pytorch_checkpoint.ckpt',
        hparams_file='/path/to/test_tube/experiment/version/hparams.yaml',
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this  case, the options you pass to trainer will be used when
running the test set (ie: 16-bit, dp, ddp, etc...)

----------

Test with additional data loaders
---------------------------------
You can still run inference on a test set even if the `test_dataloader` method hasn't been
defined within your :ref:`lightning_module` instance. This would be the case when your test data
is not available at the time your model was declared.

.. code-block:: python

    # setup your data loader
    test = DataLoader(...)

    # test (pass in the loader)
    trainer.test(test_dataloaders=test)

You can either pass in a single dataloader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases. Additionally,
you can also pass in an :ref:`datamodules` that have overridden the
:ref:`datamodule-test-dataloader-label` method.

.. code-block:: python

    class MyDataModule(pl.LightningDataModule):
        ...
        def test_dataloader(self):
            return DataLoader(...)

    # setup your datamodule
    dm = MyDataModule(...)

    # test (pass in datamodule)
    trainer.test(datamodule=dm)
