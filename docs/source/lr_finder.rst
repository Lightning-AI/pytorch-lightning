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

Using Lightnings build-in LR finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the most basic use case, this feature can be enabled during trainer construction
with ``Trainer(auto_lr_find=True)``. When ``.fit(model)`` is called, the lr finder
will automatically be run before any training is done. The ``lr`` that is found
and used will we written to the console and logged together with all other
hyperparameters of the model.

.. note:: If ``auto_lr_find=True``, it is expected that the ``hparams`` of the 
    model either has a ``lr`` or ``learning_rate`` field that can be overridden. 
    Additionally ``auto_lr_find`` can be set to a string ``s``, which will then
    try to override ``model.hparams.s``. In both cases, if the respective fields
    are not found, an error will be thrown.

If you want to inspect the results of the learning rate finder before doing any
actual training or just play around with the parameters of the algorithm, this
can be done by invoking the ``find_lr`` method of the trainer. A typical example
of this would look like

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = pl.Trainer()
    
    # Run learning rate finder
    lrfinder = trainer.find_lr(model)
    
    # Results can be found in
    lrfinder.results
    
    # Plot with
    fig = lrfinder.plot(suggest=True)
    fig.show()
    
    # Pick point based on plot, or get suggestion
    new_lr = lrfinder.suggestion()
    
    # update hparams of the model
    model.hparams.lr = new_lr
    
    # Fit model
    trainer.fit(model)
    
The figure produced by ``lrfinder.plot()`` should look something like the figure
below. It is recommended to not pick the learning rate that achives the lowest
loss, but instead something in the middle of the sharpest downward slope (red point).
This is the point returned py ``lrfinder.suggestion()``.

.. figure:: /_images/trainer/lr_finder.png

The parameters of the algorithm can be seen below.

.. autoclass:: pytorch_lightning.trainer.lr_finder.TrainerLRFinderMixin
   :members: find_lr
   :noindex:
   :exclude-members: _atomic_save, _run_lr_finder_internally, _model_dump, _model_restore