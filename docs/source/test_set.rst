Test set
========
Lightning forces the user to run the test set separately to make sure it isn't evaluated by mistake.


Test after fit
--------------
To run the test set after training completes, use this method

.. code-block:: python

    # run full training
    trainer.fit(model)

    # run test set
    trainer.test()


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


Test with additional data loaders
---------------------------------
You can still run inference on a test set even if the `test_dataloader` method hasn't been
defined within your :class:`~pytorch_lightning.core.LightningModule` instance. This would be the case when your test data
is not available at the time your model was declared.

.. code-block:: python

    # setup your data loader
    test = DataLoader(...)

    # test (pass in the loader)
    trainer.test(test_dataloaders=test)

You can either pass in a single dataloader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases.
