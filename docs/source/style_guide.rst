###########
Style Guide
###########
A main goal of Lightning is to improve readability and reproducibility. Imagine looking into any GitHub repo,
finding a lightning module and knowing exactly where to look to find the things you care about.

The goal of this style guide is to encourage Lightning code to be structured similarly.

--------------

****************************
LightningModule method order
****************************
The only required methods in the LightningModule are:

* init
* training_step
* configure_optimizers

However, if you decide to implement the rest of the optional methods, the recommended order is:

* model/system definition (init)
* if doing inference, define forward
* training hooks
* validation hooks
* test hooks
* configure_optimizers
* any other hooks

In practice, this code looks like:

.. code-block:: python

    class LitModel(pl.LightningModule):

        def __init__(...):

        def forward(...):

        def training_step(...)

        def training_step_end(...)

        def training_epoch_end(...)

        def validation_step(...)

        def validation_step_end(...)

        def validation_epoch_end(...)

        def test_step(...)

        def test_step_end(...)

        def test_epoch_end(...)

        def configure_optimizers(...)

        def any_extra_hook(...)

-----------

************************
Forward vs training_step
************************
We recommend using forward for inference/predictions and keeping training_step independent

.. code-block:: python

    def forward(...):
        embeddings = self.encoder(x)

    def training_step(...):
        x, y = ...
        z = self.encoder(x)
        pred = self.decoder(z)
        ...

However, when using DataParallel, you will need to call forward manually

.. code-block:: python

    def training_step(...):
        x, y = ...
        z = self(x)  # < ---------- instead of self.encoder(x)
        pred = self.decoder(z)
        ...
