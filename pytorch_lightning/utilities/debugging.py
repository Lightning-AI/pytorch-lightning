"""
These flags are useful to help debug a model.

Fast dev run
------------

This flag is meant for debugging a full train/val/test loop.
 It'll activate callbacks, everything but only with 1 training and 1 validation batch.
 Use this to debug a full run of your program quickly

.. code-block:: python

    # DEFAULT
    trainer = Trainer(fast_dev_run=False)


Inspect gradient norms
----------------------

Looking at grad norms can help you figure out where training might be going wrong.

.. code-block:: python

    # DEFAULT (-1 doesn't track norms)
    trainer = Trainer(track_grad_norm=-1)

    # track the LP norm (P=2 here)
    trainer = Trainer(track_grad_norm=2)


Make model overfit on subset of data
------------------------------------

A useful debugging trick is to make your model overfit a tiny fraction of the data.

setting `overfit_pct > 0` will overwrite train_percent_check, val_percent_check, test_percent_check

.. code-block:: python

    # DEFAULT don't overfit (ie: normal training)
    trainer = Trainer(overfit_pct=0.0)

    # overfit on 1% of data
    trainer = Trainer(overfit_pct=0.01)


Print the parameter count by layer
----------------------------------

By default lightning prints a list of parameters *and submodules* when it starts training.

.. code-block:: python

    # DEFAULT print a full list of all submodules and their parameters.
    trainer = Trainer(weights_summary='full')

    # only print the top-level modules (i.e. the children of LightningModule).
    trainer = Trainer(weights_summary='top')

Print which gradients are nan
-----------------------------

This option prints a list of tensors with nan gradients::

    # DEFAULT
    trainer = Trainer(print_nan_grads=False)

Log GPU usage
-------------

Lightning automatically logs gpu usage to the test tube logs.
 It'll only do it at the metric logging interval, so it doesn't slow down training.

"""


class MisconfigurationException(Exception):
    pass
