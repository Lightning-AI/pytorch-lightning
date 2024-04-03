:orphan:

.. _debugging_intermediate:


###############################
Debug your model (intermediate)
###############################
**Audience**: Users who want to debug their ML code

----

***************************
Why should I debug ML code?
***************************
Machine learning code requires debugging mathematical correctness, which is not something non-ML code has to deal with. Lightning implements a few best-practice techniques to give all users, expert level ML debugging abilities.

----

**************************************
Overfit your model on a Subset of Data
**************************************

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :paramref:`~lightning.pytorch.trainer.trainer.Trainer.overfit_batches`
argument of :class:`~lightning.pytorch.trainer.trainer.Trainer`)

.. testcode::

    # use only 1% of training data
    trainer = Trainer(overfit_batches=0.01)

    # similar, but with a fixed 10 batches
    trainer = Trainer(overfit_batches=10)

    # equivalent to
    trainer = Trainer(limit_train_batches=10, limit_val_batches=10)

Setting ``overfit_batches`` is the same as setting ``limit_train_batches`` and ``limit_val_batches`` to the same value, but in addition will also turn off shuffling in the training dataloader.


----

********************************
Look-out for exploding gradients
********************************
One major problem that plagues models is exploding gradients.
Gradient clipping is one technique that can help keep gradients from exploding.

You can keep an eye on the gradient norm by logging it in your LightningModule:

.. code-block:: python

    from lightning.pytorch.utilities import grad_norm


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)


This will plot the 2-norm of each layer to your experiment manager.
If you notice the norm is going up, there's a good chance your gradients will explode.

One technique to stop exploding gradients is to clip the gradient when the norm is above a certain threshold:

.. testcode::

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
    trainer = Trainer(gradient_clip_val=0.5)

    # clip gradients' maximum magnitude to <=0.5
    trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")

----

*************************
Detect autograd anomalies
*************************
Lightning helps you detect anomalies in the PyTorh autograd engine via PyTorch's built-in
`Anomaly Detection Context-manager <https://pytorch.org/docs/stable/autograd.html#anomaly-detection>`_.

Enable it via the **detect_anomaly** trainer argument:

.. testcode::

    trainer = Trainer(detect_anomaly=True)
