Debugging
=========
The following are flags that make debugging much easier.

Fast dev run
------------
This flag runs a "unit test" by running 1 training batch and 1 validation batch.
The point is to detect any bugs in the training/validation loop without having to wait for
a full epoch to crash.

.. code-block:: python

    trainer = pl.Trainer(unit_test=True)

Inspect gradient norms
----------------------
Logs (to a logger), the norm of each weight matrix.

.. code-block:: python

    # the 2-norm
    trainer = pl.Trainer(track_grad_norm=2)

Log GPU usage
-------------
Logs (to a logger) the GPU usage for each GPU on the master machine.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(log_gpu_memory=True)

Make model overfit on subset of data
------------------------------------

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(overfit_pct=0.01)

Print the parameter count by layer
----------------------------------
Whenever the .fit() function gets called, the Trainer will print the weights summary for the lightningModule.
To disable this behavior, turn off this flag:

(See: :ref:`trainer.weights_summary`)

.. code-block:: python

    trainer = pl.Trainer(weights_summary=None)

Print which gradients are nan
-----------------------------
Prints the tensors with nan gradients.

(See: :meth:`trainer.print_nan_grads`)

.. code-block:: python

    trainer = pl.Trainer(print_nan_grads=False)

Set the number of validation sanity steps
-----------------------------------------
Lightning runs a few steps of validation in the beginning of training.
This avoids crashing in the validation loop sometime deep into a lengthy training loop.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(nb_sanity_val_steps=5)