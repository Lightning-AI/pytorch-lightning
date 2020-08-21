"""
.. testsetup:: *
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.hypertuner.hypertuner import HyperTuner
    
.. warning::
    HyperTuner is still under active development and is not meant to be used yet!
    
The hyper tuner class can assist in tuning some parameters of your model. It is
not a general hyperparameter search class, since it relies on specific search algorithms
for optimizing specific hyperparameters. Currently the `HyperTuner` class have two
tuner algorithms implemented
    * Batch size scaling
    * Learning Rate Finder
    * n_worker searcher
    
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

"""
from pytorch_lightning.hypertuner.hypertuner import HyperTuner

__all__ = [
    'HyperTuner'
]