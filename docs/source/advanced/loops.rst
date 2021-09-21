.. _loop_customization:

Loop Customization
==================

Loop customization is an experimental feature introduced in Lightning 1.5 that enables advanced users to write new training logic or modify existing behavior in Lightning's training, evaluation, or prediction loops.
By advanced users, we mean users that are familiar with the major components under the ``Trainer`` and how they interact with the ``LightningModule``.


The training loop in Lightning
------------------------------

Every PyTorch users is familiar with the basic training loop for gradient descent optimization:

.. code-block:: python

    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            x, y = batch
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

At its core, the Lightning Trainer does not do anything different here.
It implements the same loop as shown above except that the research code stays in the LightningModule:

.. code-block:: python

    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            loss = lightning_module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

What remains in the Trainer is the loop, zero_grad, backward and optimizer step calls.
These are considered *boilerplate* and get automated by Lightning.
We refer to this as *automatic optimization*.

This optimization scheme is very general and applies to the vast majority of deep learning research.
However, the loops and optimizer calls here remain predetermined in their order and are fully controlled by the Trainer.

Loop customization now enables a new level of control where also the two remaining for loops and more can be fully changed or replaced.

Here is how the above training loop can be defined using the new Loop API:

.. code-block:: python

    class FitLoop(Loop):

        def __init__(self):
            self.epoch_loop = EpochLoop()

        def run(self):
            for epoch in range(self.trainer.max_epochs)
                self.advance()

        def advance(self):
            dataloader = lightning_module.train_dataloader()
            self.epoch_loop.run(dataloader)


    class EpochLoop(Loop):

        def run(self, dataloader):
            self.iterator = enumerate(dataloader)
            while True:
                try:
                    self.advance()
                except StopIteration:
                    break

        def advance(self):
            i, batch = next(self.iterator)
            lightning_module.training_step(i, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


Defining a loop with a class interface instead of hard-coding a raw Python for/while loop has several benefits:

1. you can have full control over the data flow through loops
2. you can add new loops and nest as many of them as they want
3. if needed, the state of a loop can be saved and resumed (more on that later)
4. new hooks can be injected at any point

and much more.
When we have a custom loop defined in class as shown above, we can attach it to the trainer.

.. code-block:: python

    fit_loop = FitLoop()

    trainer = Trainer()

    # .fit() will use this loop
    trainer.fit_loop = fit_loop

    model = ...
    trainer.fit(model)
