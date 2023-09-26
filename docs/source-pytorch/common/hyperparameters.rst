:orphan:

Configure hyperparameters from the CLI
--------------------------------------

You can use any CLI tool you want with Lightning.
For beginners, we recommend using Python's built-in argument parser.


----


ArgumentParser
^^^^^^^^^^^^^^

The :class:`~argparse.ArgumentParser` is a built-in feature in Python that let's you build CLI programs.
You can use it to make hyperparameters and other training settings available from the command line:

.. code-block:: python

    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--devices", type=int, default=2)

    # Hyperparameters for the model
    parser.add_argument("--layer_1_dim", type=int, default=128)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    # Use the parsed arguments in your program
    trainer = Trainer(devices=args.devices)
    model = MyModel(layer_1_dim=args.layer_1_dim)

This allows you to call your program like so:

.. code-block:: bash

    python trainer.py --layer_1_dim 64 --devices 1

----


LightningCLI
^^^^^^^^^^^^

Python's argument parser works well for simple use cases, but it can become cumbersome to maintain for larger projects.
For example, every time you add, change, or delete an argument from your model, you will have to add, edit, or remove the corresponding ``parser.add_argument`` code.
The :doc:`Lightning CLI <../cli/lightning_cli>` provides a seamless integration with the Trainer and LightningModule for which the CLI arguments get generated automatically for you!
