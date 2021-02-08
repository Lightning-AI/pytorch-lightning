.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _debugging:

Debugging
=========

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_debugging.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+7-+debugging_1.mp4"></video>

|

The following are flags that make debugging much easier.

----------------

fast_dev_run
------------
This flag runs a "unit test" by running n if set to ``n`` (int) else 1 if set to ``True`` training and validation batch(es).
The point is to detect any bugs in the training/validation loop without having to wait for a full epoch to crash.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.fast_dev_run`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # runs 1 train, val, test batch and program ends
    trainer = Trainer(fast_dev_run=True)

    # runs 7 train, val, test batches and program ends
    trainer = Trainer(fast_dev_run=7)

.. note::

    This argument will disable tuner, checkpoint callbacks, early stopping callbacks,
    loggers and logger callbacks like ``LearningRateLogger`` and runs for only 1 epoch.

----------------

Inspect gradient norms
----------------------
Logs (to a logger), the norm of each weight matrix.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.track_grad_norm`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # the 2-norm
    trainer = Trainer(track_grad_norm=2)

----------------

Log GPU usage
-------------
Logs (to a logger) the GPU usage for each GPU on the master machine.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_gpu_memory`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    trainer = Trainer(log_gpu_memory=True)

----------------

Make model overfit on subset of data
------------------------------------

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.overfit_batches`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # use only 1% of training data (and use the same training dataloader (with shuffle off) in val and test)
    trainer = Trainer(overfit_batches=0.01)

    # similar, but with a fixed 10 batches no matter the size of the dataset
    trainer = Trainer(overfit_batches=10)

With this flag, the train, val, and test sets will all be the same train set. We will also replace the sampler
in the training set to turn off shuffle for you.

----------------

Print a summary of your LightningModule
---------------------------------------
Whenever the ``.fit()`` function gets called, the Trainer will print the weights summary for the LightningModule.
By default it only prints the top-level modules. If you want to show all submodules in your network, use the
`'full'` option:

.. testcode::

    trainer = Trainer(weights_summary='full')

You can also display the intermediate input- and output sizes of all your layers by setting the
``example_input_array`` attribute in your LightningModule. It will print a table like this

.. code-block:: text

      | Name  | Type        | Params | In sizes  | Out sizes
    --------------------------------------------------------------
    0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
    1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
    2 | net.1 | BatchNorm1d | 1.0 K    | [10, 512] | [10, 512]

when you call ``.fit()`` on the Trainer. This can help you find bugs in the composition of your layers.

See Also:
    - :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_summary` Trainer argument
    - :class:`~pytorch_lightning.core.memory.ModelSummary`

----------------

Shorten epochs
--------------
Sometimes it's helpful to only use a percentage of your training, val or test data (or a set number of batches).
For example, you can use 20% of the training set and 1% of the validation set.

On larger datasets like Imagenet, this can help you debug or test a few things faster than waiting for a full epoch.

.. testcode::

    # use only 10% of training data and 1% of val data
    trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

    # use 10 batches of train and 5 batches of val
    trainer = Trainer(limit_train_batches=10, limit_val_batches=5)

----------------

Set the number of validation sanity steps
-----------------------------------------
Lightning runs a few steps of validation in the beginning of training.
This avoids crashing in the validation loop sometime deep into a lengthy training loop.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.num_sanity_val_steps`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=2)
