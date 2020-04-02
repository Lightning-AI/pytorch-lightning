Fast Training
=============
There are multiple options to speed up different parts of the training by choosing to train
on a subset of data. This could be done for speed or debugging purposes.

Check validation every n epochs
-------------------------------
If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)

Force training for min or max epochs
------------------------------------
It can be useful to force training for a minimum number of epochs or limit to a max number.

.. seealso::
    :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)


Set validation check frequency within 1 training epoch
------------------------------------------------------
For large datasets it's often desirable to check validation multiple times within a training loop.
Pass in a float to check that often within 1 training epoch. Pass in an int k to check every k training batches.
Must use an int if using an IterableDataset.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)

Use training data subset
------------------------
If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

.. code-block:: python

   # DEFAULT
   trainer = Trainer(train_percent_check=1.0)

   # check 10% only
   trainer = Trainer(train_percent_check=0.1)

.. note:: ``train_percent_check`` will be overwritten by ``overfit_pct`` if ``overfit_pct`` > 0.

Use test data subset
--------------------
If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(test_percent_check=1.0)

    # check 10% only
    trainer = Trainer(test_percent_check=0.1)

.. note:: ``test_percent_check`` will be overwritten by ``overfit_pct`` if ``overfit_pct`` > 0.

Use validation data subset
--------------------------
If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_percent_check=1.0)

    # check 10% only
    trainer = Trainer(val_percent_check=0.1)

.. note:: ``val_percent_check`` will be overwritten by ``overfit_pct`` if ``overfit_pct`` > 0 and ignored if
    ``fast_dev_run=True``.