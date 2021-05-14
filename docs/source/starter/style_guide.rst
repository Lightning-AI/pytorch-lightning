###########
Style guide
###########
A main goal of Lightning is to improve readability and reproducibility. Imagine looking into any GitHub repo,
finding a lightning module and knowing exactly where to look to find the things you care about.

The goal of this style guide is to encourage Lightning code to be structured similarly.

--------------

***************
LightningModule
***************
These are best practices about structuring your LightningModule

Systems vs models
=================

.. figure:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/model_system.png
    :width: 400

The main principle behind a LightningModule is that a full system should be self-contained.
In Lightning we differentiate between a system and a model.

A model is something like a resnet18, RNN, etc.

A system defines how a collection of models interact with each other. Examples of this are:

* GANs
* Seq2Seq
* BERT
* etc

A LightningModule can define both a system and a model.

Here's a LightningModule that defines a model:

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, num_layers: int = 3):
            super().__init__()
            self.layer_1 = nn.Linear()
            self.layer_2 = nn.Linear()
            self.layer_3 = nn.Linear()

Here's a LightningModule that defines a system:

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

For fast prototyping it's often useful to define all the computations in a LightningModule. For reusability
and scalability it might be better to pass in the relevant backbones.

Self-contained
==============
A Lightning module should be self-contained. A good test to see how self-contained your model is, is to ask
yourself this question:

"Can someone drop this file into a Trainer without knowing anything about the internals?"

For example, we couple the optimizer with a model because the majority of models require a specific optimizer with
a specific learning rate scheduler to work well.

Init
====
The first place where LightningModules tend to stop being self-contained is in the init. Try to define all the relevant
sensible defaults in the init so that the user doesn't have to guess.

Here's an example where a user will have to go hunt through files to figure out how to init this LightningModule.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, params):
            self.lr = params.lr
            self.coef_x = params.coef_x

Models defined as such leave you with many questions; what is coef_x? is it a string? a float? what is the range? etc...

Instead, be explicit in your init

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, encoder: nn.Module, coeff_x: float = 0.2, lr: float = 1e-3):
            ...

Now the user doesn't have to guess. Instead they know the value type and the model has a sensible default where the
user can see the value immediately.


Method order
============
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

Forward vs training_step
========================
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

--------------

****
Data
****
These are best practices for handling data.

Dataloaders
===========
Lightning uses dataloaders to handle all the data flow through the system. Whenever you structure dataloaders,
make sure to tune the number of workers for maximum efficiency.

.. warning:: Make sure not to use ddp_spawn with num_workers > 0 or you will bottleneck your code.

DataModules
===========
Lightning introduced datamodules. The problem with dataloaders is that sharing full datasets is often still challenging
because all these questions need to be answered:

* What splits were used?
* How many samples does this dataset have?
* What transforms were used?
* etc...

It's for this reason that we recommend you use datamodules. This is specially important when collaborating because
it will save your team a lot of time as well.

All they need to do is drop a datamodule into a lightning trainer and not worry about what was done to the data.

This is true for both academic and corporate settings where data cleaning and ad-hoc instructions slow down the progress
of iterating through ideas.
