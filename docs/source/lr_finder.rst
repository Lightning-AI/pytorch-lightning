Learning Rate Finder
--------------------

For training deep neural networks, selecting a good learning rate is essential
for both better performance and faster convergence. Even optimizers such as
`Adam` that are self-adjusting the learning rate can benefit from more optimal
choices.

To reduce the amount of guesswork concerning choosing a good initial learning
rate, a `learning rate finder` can be used. As described in this `paper <https://arxiv.org/abs/1506.01186>`_ 
a learning rate finder does a small run where the learning rate is increased 
after each processed batch and the corresponding loss is logged. The result of 
this is a `lr` vs. `loss` plot that can be used as guidence for choosing a optimal
initial lr. 

.. warning:: For the moment, this feature only works with models having a single optimizer.

Using Lightnings build-in LR finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the most basic use case, this feature can be enabled during trainer construction
with ``Trainer(auto_lr_find=True)``. When ``.fit(model)`` is called, the lr finder
will automatically be run before any training is done. The ``lr`` that is found
and used will be written to the console and logged together with all other
hyperparameters of the model.
    
.. code-block:: python
        
    # default, no automatic learning rate finder
    Trainer(auto_lr_find=True)

When the ``lr`` or ``learning_rate`` key in hparams exists, this flag sets your learning_rate.
In both cases, if the respective fields are not found, an error will be thrown.
        
.. code-block:: python

    class LitModel(LightningModule):
        def __init__(self, hparams):
            self.hparams = hparams

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=self.hparams.lr|self.hparams.learning_rate)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    Trainer(auto_lr_find=True)

To use an arbitrary value set it in the parameter.

.. code-block:: python

    # to set to your own hparams.my_value
    Trainer(auto_lr_find='my_value')

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
    trainer = pl.Trainer()
    
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

.. autoclass:: pytorch_lightning.trainer.lr_finder.TrainerLRFinderMixin
   :members: lr_find
   :noindex:
   :exclude-members: _run_lr_finder_internally, save_checkpoint, restore
