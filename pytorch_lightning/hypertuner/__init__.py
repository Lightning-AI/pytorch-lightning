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

.. warning::

    HyperTuner is not supported for DDP or 16-bit precision yet,
    it is coming soon.

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

Both methods return a single object that can be used to investigate the results
afterwards. Each object comes with the following fields/methods

* `obj.results`: dict with the information logged from the search
* `fig = obj.plot(...)`: method for plotting the results of the search
* `new_val = obj.suggestion(...)`: method for getting suggestion for optimal value to use

----------

Auto scaling of batch size
==========================

Auto scaling of batch size may be enabled to find the largest batch size that fits into
memory. Larger batch size often yields better estimates of gradients, but may also result in
longer training time.

Currently, this feature supports two search modes: `'power'` scaling and `'binsearch'`
scaling. In `'power'` scaling (default), starting from a batch size of 1 keeps doubling
the batch size until an out-of-memory (OOM) error is encountered. Setting the
argument to `'binsearch'` continues to finetune the batch size by performing
a binary search.

.. note::

    This feature does *NOT* work when passing dataloaders directly to `.fit()`. This
    is due to your dataloader needs to depend on field `batch_size` that can be adjusted
    during the search i.e. your `train_dataloader()` should look something like this

    .. code-block:: python

        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size)

The scaling algorithm has a number of parameters that the user can control by
invoking the tuner method `.scale_batch_size` themself (see description below).

.. code-block:: python

    # Construct trainer and default hypertuner
    trainer = Trainer(...)
    tuner = HyperTuner(trianer, ...)

    # Invoke method
    batch_size_obj = tuner.scale_batch_size(model, ...)

    # Plot results
    batch_size_obj.plot(suggest=True, show=True)

    # Get suggestion
    new_batch_size = batch_size_obj.suggestion()

    # Override old batch size
    model.hparams.batch_size = new_batch_size

    # Fit as normal
    trainer.fit(model)

.. autoclass:: pytorch_lightning.hypertuner.batch_scaler.HyperTunerBatchScalerMixin
   :members: scale_batch_size
   :noindex:

The `.scale_batch_size` method returns a simple object that can be used to
interact with the results.

.. autoclass:: pytorch_lightning.hypertuner.batch_scaler.BatchScaler
    :members: plot, suggestion
    :noindex:

----------

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

.. note:: For the moment, this feature only works with models having a single optimizer.

This feature supports both monitoring the learning rate vs the training loss and
monitoring the learning rate vs the validation loss. While monitoring the validation
loss normally give better estimates of the learning rate than monitoring the training
loss, it comes at the severe computationally expense since we have to evaluate the
full validation set after each gradient step.

The parameters of the learning rate finder can be adjusted by invoking the
tuner method `lr_find`. A typical example of this would look like

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = Trainer()
    tuner = HyperTuner(trainer)

    # Run learning rate finder
    lr_finder = tuner.lr_find(model)

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

The `.lr_find` method returns a simple object that can be used to
interact with the results.

.. autoclass:: pytorch_lightning.hypertuner.lr_finder.LRFinderCallback
    :members: plot, suggestion
    :noindex:

"""

from pytorch_lightning.hypertuner.hypertuner import HyperTuner
