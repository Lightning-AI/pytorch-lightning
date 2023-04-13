:orphan:

##############################
Multiple Models and Optimizers
##############################

Fabric makes it very easy to work with multiple models and/or optimizers at once in your training workflow.
Examples of where this comes in handy are Generative Adversarial Networks (GANs), Auto-encoders, meta-learning and more.


----

************************
One model, one optimizer
************************

Fabric has a simple guideline you should follow:
If you have an optimizer, you should set it up together with the model to make your code truly strategy-agnostic.

.. code-block:: python

    import torch
    from lightning.fabric import Fabric

    fabric = Fabric()

    # Instantiate model and optimizer
    model = LitModel()
    optimizer = torch.optim.Adam(model.parameters())

    # Set up the model and optimizer together
    model, optimizer = fabric.setup(model, optimizer)


Depending on the selected strategy, the :meth:`~lightning.fabric.fabric.Fabric.setup` method will wrap and link the model with the optimizer.


----


******************************
One model, multiple optimizers
******************************

You can also have multiple optimizers over a single model.
This is useful if you need specific optimizers or learning rates for parts of the model.

.. code-block:: python

    # Instantiate model and optimizers
    model = LitModel()
    optimizer1 = torch.optim.SGD(model.layer1.parameters(), lr=0.003)
    optimizer2 = torch.optim.SGD(model.layer2.parameters(), lr=0.01)

    # Set up the model and optimizers together
    model, optimizer1, optimizer2 = fabric.setup(model, optimizer1, optimizer2)



----


******************************
Multiple models, one optimizer
******************************

Using a single optimizer to update multiple models is possible too.
The best way to do this is to group all your individual models under one top level ``nn.Module``:

.. code-block:: python

    class AutoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Group all models under a common nn.Module
            self.encoder = Encoder()
            self.decoder = Decoder()

Now all of these models can be treated as a single one:

.. code-block:: python

    # Instantiate the big model
    autoencoder = AutoEncoder()
    optimizer = ...

    # Set up the model(s) and optimizer together
    autoencoder, optimizer = fabric.setup(autoencoder, optimizer)


----


************************************
Multiple models, multiple optimizers
************************************

You can pair up as many models and optimizers as you want. For example, two models with one optimizer each:

.. code-block:: python

    # Two models
    generator = Generator()
    discriminator = Discriminator()

    # Two optimizers
    optimizer_gen = torch.optim.SGD(generator.parameters(), lr=0.01)
    optimizer_dis = torch.optim.SGD(discriminator.parameters(), lr=0.001)

    # Set up generator
    generator, optimizer_gen = fabric.setup(generator, optimizer_gen)
    # Set up discriminator
    discriminator, optimizer_dis = fabric.setup(discriminator, optimizer_dis)

For a full example of this use case, see our `GAN example <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/dcgan>`_.
