Test set
========
Lightning forces the user to run the test set separately to make sure it isn't evaluated by mistake


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