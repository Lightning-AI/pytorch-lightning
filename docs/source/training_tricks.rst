.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer


Training Tricks
================
Lightning implements various tricks to help during training

Accumulate gradients
-------------------------------------
Accumulated gradients runs K small batches of size N before doing a backwards pass.
The effect is a large effective batch size of size KxN.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)


Gradient Clipping
-------------------------------------
Gradient clipping may be enabled to avoid exploding gradients. Specifically, this will `clip the gradient
norm <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_ computed over all model parameters together.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)

Auto scaling of batch size
--------------------------
Auto scaling of batch size may be enabled to find the largest batch size that fits into
memory. Larger batch size often yields better estimates of gradients, but may also result in
longer training time.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. code-block:: python

    # DEFAULT (ie: don't scale batch size automatically)
    trainer = Trainer(auto_scale_batch_size=False)

    # Autoscale batch size 
    trainer = Trainer(auto_scale_batch_size=True|'power'|'binsearch')

Setting the feature to `True` enables `'power'` scaling, that starting from a
batch size of 1 keeps doubling the batch size until an out-of-memory (OOM) error is
encountered. Setting the argument to `'binsearch'` continues to finetune the batch
size by performing a binary search. 

.. note:: 

    This feature expects that a `batch_size` field in the `hparams` of your model, i.e.,
    `model.hparams.batch_size` should exist and will be overridden by the results of this
    algorithm.

The scaling algorithm has a number of parameters that the user can control by
invoking the trainer method `.scale_batch_size` themself.

.. code-block:: python

    # Use default in trainer construction
    trainer = Trainer()

    # Invoke method
    new_batch_size = trainer.scale_batch_size(model, ...)

    # Override old batch size
    model.hparams.batch_size = new_batch_size
    
    # Fit as normal
    trainer.fit(model)

.. autoclass:: pytorch_lightning.trainer.training_tricks.TrainerTrainingTricksMixin
   :members: scale_batch_size
   :noindex:
