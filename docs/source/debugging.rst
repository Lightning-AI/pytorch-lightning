.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

Debugging
=========
The following are flags that make debugging much easier.

Fast dev run
------------
This flag runs a "unit test" by running 1 training batch and 1 validation batch.
The point is to detect any bugs in the training/validation loop without having to wait for
a full epoch to crash.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.fast_dev_run`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    trainer = Trainer(fast_dev_run=True)

Inspect gradient norms
----------------------
Logs (to a logger), the norm of each weight matrix.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.track_grad_norm`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # the 2-norm
    trainer = Trainer(track_grad_norm=2)

Log GPU usage
-------------
Logs (to a logger) the GPU usage for each GPU on the master machine.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_gpu_memory`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    trainer = Trainer(log_gpu_memory=True)

Make model overfit on subset of data
------------------------------------

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.overfit_pct`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    trainer = Trainer(overfit_pct=0.01)

Print the parameter count by layer
----------------------------------
Whenever the .fit() function gets called, the Trainer will print the weights summary for the lightningModule.
To disable this behavior, turn off this flag:

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_summary`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    trainer = Trainer(weights_summary=None)


Set the number of validation sanity steps
-----------------------------------------
Lightning runs a few steps of validation in the beginning of training.
This avoids crashing in the validation loop sometime deep into a lengthy training loop.

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.num_sanity_val_steps`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=5)