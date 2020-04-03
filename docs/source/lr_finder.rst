Learning Rate Finder
--------------------

For training deep neural networks, selecting a good learning rate is essential
for both better performance and faster convergence. Even optimizers such as
`Adam` that are self-adjusting the learning rate can benefit from more optimal
choices.

To reduce the amount of guesswork conserning choosing a good initial learning
rate, a learning rate finder can be used. As described in this (paper)[https://arxiv.org/abs/1506.01186], 
a lr finder does a small run where the lr is increased after each processed
batch and the corresponding loss is logged. The result of this is therefore a
`lr` vs. `loss` plot that can be used as guidence for choosing a optimal
initial lr. 

Using Lightnings build in LR finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical use of the build in lr finder would look something like this. It is not
recommended to pick the lr with the lowest loss, but instead choosing a point 
point corresponds to 

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = pl.Trainer()
    
    # Run learning rate finder
    lrfinder = trainer.find_lr(model)
    
    # Results can be found in
    lrfinder.results
    
    # Plot with
    lrfinder.plot(suggest=True)
    
    # Pick point based on plot, or get suggestion
    new_lr = lrfinder.suggestion()
    
    # create new model with suggested lr
    hparams.lr = new_lr
    model = MyModelClass(hparams)
    
    # Fit model
    trainer.fit(model)
    
The figure produced by `lrfinder.plot()` should look something like the figure
below. It is recommended to not pick the learning rate that achives the lowest
loss, but instead something in the middle of the sharpest downward slope (red point).
This is the point returned py `lrfinder.suggestion()`.


.. figure:: /_images/trainer/lr_finder.png