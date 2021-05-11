.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _training_tricks:

Training Tricks
================
Lightning implements various tricks to help during training

----------

Accumulate gradients
--------------------
Accumulated gradients runs K small batches of size N before doing a backwards pass.
The effect is a large effective batch size of size KxN.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

----------

Gradient Clipping
-----------------
Gradient clipping may be enabled to avoid exploding gradients. By default, this will clip the gradient norm by calling
:func:`torch.nn.utils.clip_grad_norm_` computed over all model parameters together.
If the Trainer's ``gradient_clip_algorithm`` is set to ``'value'`` (``'norm'`` by default), this will use instead
:func:`torch.nn.utils.clip_grad_norm_` for each parameter instead.

.. note::
    If using mixed precision, the ``gradient_clip_val`` does not need to be changed as the gradients are unscaled
    before applying the clipping function.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients' global norm to <=0.5
    trainer = Trainer(gradient_clip_val=0.5)  # gradient_clip_algorithm='norm' by default

    # clip gradients' maximum magnitude to <=0.5
    trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm='value')

----------

Stochastic Weight Averaging
---------------------------
Stochastic Weight Averaging (SWA) can make your models generalize better at virtually no additional cost.
This can be used with both non-trained and trained models. The SWA procedure smooths the loss landscape thus making
it harder to end up in a local minimum during optimization.

For a more detailed explanation of SWA and how it works,
read `this <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`__ post by the PyTorch team.

.. seealso:: :class:`~pytorch_lightning.callbacks.StochasticWeightAveraging` (Callback)

.. testcode::

    # Enable Stochastic Weight Averaging
    trainer = Trainer(stochastic_weight_avg=True)

----------

Auto scaling of batch size
--------------------------
Auto scaling of batch size may be enabled to find the largest batch size that fits into
memory. Larger batch size often yields better estimates of gradients, but may also result in
longer training time. Inspired by https://github.com/BlackHC/toma.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. code-block:: python

    # DEFAULT (ie: don't scale batch size automatically)
    trainer = Trainer(auto_scale_batch_size=None)

    # Autoscale batch size
    trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch')

    # find the batch size
    trainer.tune(model)

Currently, this feature supports two modes `'power'` scaling and `'binsearch'`
scaling. In `'power'` scaling, starting from a batch size of 1 keeps doubling
the batch size until an out-of-memory (OOM) error is encountered. Setting the
argument to `'binsearch'` will initially also try doubling the batch size until
it encounters an OOM, after which it will do a binary search that will finetune the
batch size. Additionally, it should be noted that the batch size scaler cannot
search for batch sizes larger than the size of the training dataset.


.. note::

    This feature expects that a `batch_size` field is either located as a model attribute
    i.e. `model.batch_size` or as a field in your `hparams` i.e. `model.hparams.batch_size`.
    The field should exist and will be overridden by the results of this algorithm.
    Additionally, your `train_dataloader()` method should depend on this field
    for this feature to work i.e.

    .. code-block:: python

        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size|self.hparams.batch_size)

.. warning::

    Due to these constraints, this features does *NOT* work when passing dataloaders directly
    to `.fit()`.

The scaling algorithm has a number of parameters that the user can control by
invoking the :meth:`~pytorch_lightning.tuner.tuning.Tuner.scale_batch_size` method:

.. code-block:: python

    # Use default in trainer construction
    trainer = Trainer()
    tuner = Tuner(trainer)

    # Invoke method
    new_batch_size = tuner.scale_batch_size(model, *extra_parameters_here)

    # Override old batch size (this is done automatically)
    model.hparams.batch_size = new_batch_size

    # Fit as normal
    trainer.fit(model)

The algorithm in short works by:
    1. Dumping the current state of the model and trainer
    2. Iteratively until convergence or maximum number of tries `max_trials` (default 25) has been reached:
        - Call `fit()` method of trainer. This evaluates `steps_per_trial` (default 3) number of
          training steps. Each training step can trigger an OOM error if the tensors
          (training batch, weights, gradients, etc.) allocated during the steps have a
          too large memory footprint.
        - If an OOM error is encountered, decrease batch size else increase it.
          How much the batch size is increased/decreased is determined by the chosen
          strategy.
    3. The found batch size is saved to either `model.batch_size` or `model.hparams.batch_size`
    4. Restore the initial state of model and trainer

.. warning:: Batch size finder is not supported for DDP yet, it is coming soon.


Advanced GPU Optimizations
--------------------------

When training on single or multiple GPU machines, Lightning offers a host of advanced optimizations to improve throughput, memory efficiency, and model scaling.
Refer to :doc:`Advanced GPU Optimized Training for more details <../advanced/advanced_gpu>`.
