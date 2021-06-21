.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _lr_finder:

Learning Rate Finder
--------------------

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/auto_lr_find.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/auto_lr_find.mp4"></video>

|

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

.. warning::
    For the moment, this feature only works with models having a single optimizer.
    LR Finder support for DDP is not implemented yet, it is coming soon.

----------

Using Lightning's built-in LR finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable the learning rate finder, your :doc:`lightning module <../common/lightning_module>` needs to have a ``learning_rate`` or ``lr`` property.
Then, set ``Trainer(auto_lr_find=True)`` during trainer construction,
and then call ``trainer.tune(model)`` to run the LR finder. The suggested ``learning_rate``
will be written to the console and will be automatically set to your :doc:`lightning module <../common/lightning_module>`,
which can be accessed via ``self.learning_rate`` or ``self.lr``.

.. code-block:: python

    class LitModel(LightningModule):

        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    model = LitModel()

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer = Trainer(auto_lr_find=True)

    trainer.tune(model)

If your model is using an arbitrary value instead of ``self.lr`` or ``self.learning_rate``, set that value as ``auto_lr_find``:

.. code-block:: python

    model = LitModel()

    # to set to your own hparams.my_value
    trainer = Trainer(auto_lr_find='my_value')

    trainer.tune(model)


You can also inspect the results of the learning rate finder or just play around
with the parameters of the algorithm. This can be done by invoking the
:meth:`~pytorch_lightning.tuner.tuning.Tuner.lr_find` method. A typical example of this would look like:

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = Trainer()

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)

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
below. It is recommended to not pick the learning rate that achieves the lowest
loss, but instead something in the middle of the sharpest downward slope (red point).
This is the point returned py ``lr_finder.suggestion()``.

.. figure:: ../_static/images/trainer/lr_finder.png

The parameters of the algorithm can be seen below.

.. autofunction:: pytorch_lightning.tuner.lr_finder.lr_find
   :noindex:
