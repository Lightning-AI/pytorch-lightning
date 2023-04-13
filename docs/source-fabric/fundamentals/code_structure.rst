######################################
How to structure your code with Fabric
######################################

Fabric is flexible enough to adapt to any project structure, regardless of whether you are experimenting with a simple script or an extensive framework, because it makes no assumptions about how your code is organized.
Despite the ultimate freedom, this page is meant to give beginners a template for how to organize a typical training script with Fabric:
We also have several :doc:`examples <../examples/index>` that you can take inspiration from.


----


*****************
The Main Function
*****************

At the highest level, every Python script should contain the following boilerplate code to guard the entry point for the main function:

.. code-block:: python

    def main():
        # Here goes all the rest of the code
        ...


    if __name__ == "__main__":
        # This is the entry point of your program
        main()


This ensures that any form of multiprocessing will work properly (for example, ``DataLoader(num_workers=...)`` etc.)


----


**************
Model Training
**************

Here is a skeleton for training a model in a function ``train()``:

.. code-block:: python

    import lightning as L


    def train(fabric, model, optimizer, dataloader):
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                ...


    def main():
        # (Optional) Parse command line options
        args = parse_args()

        # Configure Fabric
        fabric = L.Fabric(...)

        # Instantiate objects
        model = ...
        optimizer = ...
        train_dataloader = ...

        # Set up objects
        model, optimizer = fabric.setup(model, optimizer)
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

        # Run training loop
        train(fabric, model, optimizer, train_dataloader)


    if __name__ == "__main__":
        main()


----


*****************************
Training, Validation, Testing
*****************************

Often it is desired to evaluate the ability of the model to generalize on unseen data.
Here is how the code would be structured if we did that periodically during training (called validation) and after training (called testing).


.. code-block:: python

    import lightning as L


    def train(fabric, model, optimizer, train_dataloader, val_dataloader):
        # Training loop with validation every few epochs
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_dataloader):
                ...

            if epoch % validate_every_n_epoch == 0:
                validate(fabric, model, val_dataloader)


    def validate(fabric, model, dataloader):
        # Validation loop
        model.eval()
        for i, batch in enumerate(dataloader):
            ...


    def test(fabric, model, dataloader):
        # Test/Prediction loop
        model.eval()
        for i, batch in enumerate(dataloader):
            ...


    def main():
        ...

        # Run training loop with validation
        train(fabric, model, optimizer, train_dataloader, val_dataloader)

        # Test on unseen data
        test(fabric, model, test_dataloader)


    if __name__ == "__main__":
        main()



----


************
Full Trainer
************

Building a fully-fledged, personalized Trainer can be a lot of work.
To get started quickly, copy `this <https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer>`_ Trainer template and adapt it to your needs.

- Only ~500 lines of code, all in one file
- Relies on Fabric to configure accelerator, devices, strategy
- Simple epoch based training with validation loop
- Only essential features included: Checkpointing, loggers, progress bar, callbacks, gradient accumulation


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Trainer Template
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>
