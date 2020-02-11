Training Tricks
================
Lightning implements various tricks to help during training

Accumulate gradients
-------------------------------------
Accumulated gradients runs K small batches of size N before doing a backwards pass.
The effect is a large effective batch size of size KxN.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)


Gradient Clipping
-------------------------------------
Gradient clipping may be enabled to avoid exploding gradients. Specifically, this will `clip the gradient
norm <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_ computed over all model parameters together.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)


Set how much of the training set to check (1-100%)
---------------------------------------------------
If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

.. code-block:: python

   # DEFAULT
   trainer = Trainer(train_percent_check=1.0)

   # check 10% only
   trainer = Trainer(train_percent_check=0.1)

.. note:: train_percent_check will be overwritten by overfit_pct if overfit_pct > 0

Force training for min or max epochs
-------------------------------------
It can be useful to force training for a minimum number of epochs or limit to a max number.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)


Validation loop
================

Check validation every n epochs
-------------------------------------
If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)


Set how much of the validation set to check
--------------------------------------------
If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag
val_percent_check will be overwritten by overfit_pct if overfit_pct > 0

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_percent_check=1.0)

    # check 10% only
    trainer = Trainer(val_percent_check=0.1)


Set how much of the test set to check
-------------------------------------
If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag
test_percent_check will be overwritten by overfit_pct if overfit_pct > 0.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(test_percent_check=1.0)

    # check 10% only
    trainer = Trainer(test_percent_check=0.1)


Set validation check frequency within 1 training epoch
-------------------------------------
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