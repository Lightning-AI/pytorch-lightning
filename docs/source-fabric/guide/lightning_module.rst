##################
Organize Your Code
##################

Any raw PyTorch can be converted to Fabric with zero refactoring required, giving maximum flexibility in how you want to organize your projects.

However, when developing a project in a team or sharing the code publicly, it can be beneficial to conform to a standard format of how core pieces of the code are organized.
This is what the `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_ was made for!

Here is how you can neatly separate the research code (model, loss, optimization, etc.) from the "trainer" code (training loop, checkpointing, logging, etc.).


----


*************************************************
Step 1: Move your code into LightningModule hooks
*************************************************

Take these main ingredients and put them in a LightningModule:

- The PyTorch model(s) as an attribute (e.g. ``self.model``)
- The forward, including loss computation, goes into ``training_step()``
- Setup of optimizer(s) goes into ``configure_optimizers()``
- Setup of the training data loader goes into ``train_dataloader()``


.. code-block:: python

    import lightning as L


    class LitModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = ...

        def training_step(self, batch, batch_idx):
            # Main forward, loss computation, and metrics goes here
            x, y = batch
            y_hat = self.model(x)
            loss = self.loss_fn(y, y_hat)
            acc = self.accuracy(y, y_hat)
            ...
            return loss

        def configure_optimizers(self):
            # Return one or several optimizers
            return torch.optim.Adam(self.parameters(), ...)

        def train_dataloader(self):
            # Return your dataloader for training
            return DataLoader(...)

        def on_train_start(self):
            # Do something at the beginning of training
            ...

        def any_hook_you_like(self, *args, **kwargs):
            ...


This is a minimal LightningModule, but there are `many other useful hooks <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks>`_ you can use.


----


****************************************
Step 2: Call hooks from your Fabric code
****************************************

In your Fabric training loop, you can now call the hooks of the LightningModule interface.
It is up to you to call everything at the right place.

.. code-block:: python

    import lightning as L

    fabric = L.Fabric(...)

    # Instantiate the LightningModule
    model = LitModel()

    # Get the optimizer(s) from the LightningModule
    optimizer = model.configure_optimizers()

    # Get the training data loader from the LightningModule
    train_dataloader = model.train_dataloader()

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Call the hooks at the right time
    model.on_train_start()

    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, i)
            fabric.backward(loss)
            optimizer.step()

            # Control when hooks are called
            if condition:
                model.any_hook_you_like()


Your code is now modular. You can switch out the entire LightningModule implementation for another one, and you don't need to touch the training loop:

.. code-block:: diff

    # Instantiate the LightningModule
  - model = LitModel()
  + model = DopeModel()

    ...


----


************************************
Access Fabric inside LightningModule
************************************

You can access the Fabric instance in any of the LightningModule hooks via ``self.fabric``, provided that you called
``fabric.setup()`` on the module.

.. code-block:: python

    import lightning as L


    class LitModel(L.LightningModule):
        def on_train_start(self):
            # Access Fabric and its attributes
            print(self.fabric.world_size)


    fabric = L.Fabric()
    model = fabric.setup(LitModel())
    model.on_train_start()

To maximize compatibility with LightningModules written for the Lightning Trainer, ``self.trainer`` is also available and will
reroute to ``self.fabric``.
