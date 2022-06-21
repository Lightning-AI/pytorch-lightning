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

(See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.overfit_batches`
argument of :class:`~pytorch_lightning.trainer.trainer.Trainer`)

.. testcode::

    # use only 1% of training data (and turn off validation)
    trainer = Trainer(overfit_batches=0.01)

    # similar, but with a fixed 10 batches
    trainer = Trainer(overfit_batches=10)

When using this argument, the validation loop will be disabled. We will also replace the sampler
in the training set to turn off shuffle for you.

----

********************************
Look-out for exploding gradients
********************************
One major problem that plagues models is exploding gradients. Gradient norm is one technique that can help keep gradients from exploding.

.. testcode::

    # the 2-norm
    trainer = Trainer(track_grad_norm=2)

This will plot the 2-norm to your experiment manager. If you notice the norm is going up, there's a good chance your gradients are/will explode.

One technique to stop exploding gradients is to clip the gradient

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
