"""
.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.hypertuner.hypertuner import HyperTuner


The hyper tuner class can assist in tuning some parameters of your model. It is
not a general hyperparameter search class, since it relies on specific search algorithms
for optimizing specific hyperparameters. Currently the `HyperTuner` class have two
tuner algorithms implemented

    * Batch size scaling
    * Learning Rate Finder

.. note::
    
    The `HyperTuner` class is **experimental**. This means that some of its
    functionality is under tested and its interface *may* change drastically
    within the next few releases
    
*************************************
Automatic hyperparameter optimization
*************************************

Most users should be able to use the `HyperTuner` class with their existing
lightning implementation to automatically optimize some of their hyperparameters.
This can be done by:
    
.. code-block:: python
    
    from pytorch_lightning import Trainer, HyperTuner
    
    # Instanciate model and trainer
    model = ModelClass(...)
    trainer = Trainer(...)
    
    # Automatically tune hyperparameters
    tuner = HyperTuner(trainer, 
                       auto_scale_batch_size=True,
                       auto_lr_find=True)
    tuner.tune(model)  # automatically tunes hyperparameters

    # Fit as normally
    trainer.fit(model)

The main method of the `HyperTuner` class is the `.tune` method. This method
works similar to `.fit` of the trainer class. This will automatically run
the hyperparameter search using default search parameters. 

.. autoclass:: pytorch_lightning.hypertuner.hypertuner.HyperTuner
   :members: tune
   :noindex:
   :exclude-members: _call_internally

The `.tune` method assumes that your model have a field where the results can be
written to. For example, if `auto_scale_batch_size=True` the results will be tried
written to either (in this order):
    * model.batch_size
    * model.hparams.batch_size
    * model.hparams['batch_size']
and throw an error if not able to. If you instead want to write to another field
you can specify this with a string: `auto_scale_batch_size='my_batch_size_field'`.
This works simiarly for the `auto_lr_find` argument.

***************
Tuner algoritms
***************

The default search strategy may not be optimal for your specific model
and the individual algorithms can therefore be invoked using the `HyperTuner`
class to gain more control over the search.

Auto scaling of batch size
==========================
Auto scaling of batch size may be enabled to find the largest batch size that fits into
memory. Larger batch size often yields better estimates of gradients, but may also result in
longer training time.

Currently, this feature supports two modes `'power'` scaling and `'binsearch'`
scaling. In `'power'` scaling, starting from a batch size of 1 keeps doubling
the batch size until an out-of-memory (OOM) error is encountered. Setting the
argument to `'binsearch'` continues to finetune the batch size by performing
a binary search.

.. note::

    This feature expects that a `batch_size` field in the `hparams` of your model, i.e.,
    `model.hparams.batch_size` should exist and will be overridden by the results of this
    algorithm. Additionally, your `train_dataloader()` method should depend on this field
    for this feature to work i.e.

    .. code-block:: python

        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size)

.. warning::

    Due to these constraints, this features does *NOT* work when passing dataloaders directly
    to `.fit()`.

The scaling algorithm has a number of parameters that the user can control by
invoking the trainer method `.scale_batch_size` themself (see description below).

.. code-block:: python

    # Use default in trainer construction
    trainer = Trainer()

    # Invoke method
    new_batch_size = trainer.scale_batch_size(model, ...)

    # Override old batch size
    model.hparams.batch_size = new_batch_size

    # Fit as normal
    trainer.fit(model)

The algorithm in short works by:
    1. Dumping the current state of the model and trainer
    2. Iteratively until convergence or maximum number of tries `max_trials` (default 25) has been reached:
        - Call `fit()` method of trainer. This evaluates `steps_per_trial` (default 3) number of
          training steps. Each training step can trigger an OOM error if the tensors
          (training batch, weights, gradients ect.) allocated during the steps have a
          too large memory footprint.
        - If an OOM error is encountered, decrease batch size else increase it.
          How much the batch size is increased/decreased is determined by the choosen
          stratrgy.
    3. The found batch size is saved to `model.hparams.batch_size`
    4. Restore the initial state of model and trainer

.. autoclass:: pytorch_lightning.hypertuner.batch_scaler.HyperTunerBatchScalerMixin
   :members: scale_batch_size
   :noindex:

.. warning:: Batch size finder is not supported for DDP yet, it is coming soon.


Learning Rate Finder
====================

For training deep neural networks, selecting a good learning rate is essential
for both better performance and faster convergence. Even optimizers such as
`Adam` that are self-adjusting the learning rate can benefit from more optimal
choices.

To reduce the amount of guesswork concerning choosing a good initial learning
rate, a `learning rate finder` can be used. As described in this `paper <https://arxiv.org/abs/1506.01186>`_
a learning rate finder does a small run where the learning rate is increased
after each processed batch and the corresponding loss is logged. The result of
this is a `lr` vs. `loss` plot that can be used as guidance for choosing a optimal
initial lr.

Warnings:
- For the moment, this feature only works with models having a single optimizer.
- LR support for DDP is not implemented yet, it is comming soon.

Using Lightning's built-in LR finder
------------------------------------

In the most basic use case, this feature can be enabled during trainer construction
with ``Trainer(auto_lr_find=True)``. When ``.fit(model)`` is called, the LR finder
will automatically be run before any training is done. The ``lr`` that is found
and used will be written to the console and logged together with all other
hyperparameters of the model.

.. testcode::

    # default: no automatic learning rate finder
    trainer = Trainer(auto_lr_find=False)

This flag sets your learning rate which can be accessed via ``self.lr`` or ``self.learning_rate``.

.. testcode::

    class LitModel(LightningModule):

        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer = Trainer(auto_lr_find=True)

To use an arbitrary value set it in the parameter.

.. testcode::

    # to set to your own hparams.my_value
    trainer = Trainer(auto_lr_find='my_value')

Under the hood, when you call fit, this is what happens.

1. Run learning rate finder.
2. Run actual fit.

.. code-block:: python

    # when you call .fit() this happens
    # 1. find learning rate
    # 2. actually run fit
    trainer.fit(model)

If you want to inspect the results of the learning rate finder before doing any
actual training or just play around with the parameters of the algorithm, this
can be done by invoking the ``lr_find`` method of the trainer. A typical example
of this would look like

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = Trainer()

    # Run learning rate finder
    lr_finder = trainer.lr_find(model)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr

    # Fit model
    trainer.fit(model)

The figure produced by ``lr_finder.plot()`` should look something like the figure
below. It is recommended to not pick the learning rate that achives the lowest
loss, but instead something in the middle of the sharpest downward slope (red point).
This is the point returned py ``lr_finder.suggestion()``.

.. figure:: /_images/trainer/lr_finder.png

The parameters of the algorithm can be seen below.

.. autoclass:: pytorch_lightning.hypertuner.lr_finder.HyperTunerLRFinderMixin
   :members: lr_find
   :noindex:
   :exclude-members: _run_lr_finder_internally, save_checkpoint, restore


"""

from pytorch_lightning.hypertuner.hypertuner import HyperTuner
