:orphan:

#####################################################
Configure hyperparameters from the CLI (Intermediate)
#####################################################
**Audience:** Users who want advanced modularity via a command line interface (CLI).

**Pre-reqs:** You must already understand how to use the command line and :doc:`LightningDataModule <../data/datamodule>`.

----

*************************
LightningCLI requirements
*************************

The :class:`~lightning.pytorch.cli.LightningCLI` class is designed to significantly ease the implementation of CLIs. To
use this class, an additional Python requirement is necessary than the minimal installation of Lightning provides. To
enable, either install all extras:

.. code:: bash

    pip install "lightning[pytorch-extra]"

or if only interested in ``LightningCLI``, just install jsonargparse:

.. code:: bash

    pip install "jsonargparse[signatures]"

----

******************
Implementing a CLI
******************
Implementing a CLI is as simple as instantiating a :class:`~lightning.pytorch.cli.LightningCLI` object giving as
arguments classes for a ``LightningModule`` and optionally a ``LightningDataModule``:

.. code:: python

    # main.py
    from lightning.pytorch.cli import LightningCLI

    # simple demo classes for your convenience
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


    def cli_main():
        cli = LightningCLI(DemoModel, BoringDataModule)
        # note: don't call fit!!


    if __name__ == "__main__":
        cli_main()
        # note: it is good practice to implement the CLI in a function and call it in the main if block

Now your model can be managed via the CLI. To see the available commands type:

.. code:: bash

    $ python main.py --help

which prints out:

.. code:: bash

    usage: main.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
            {fit,validate,test,predict} ...

    Lightning Trainer command line tool

    optional arguments:
    -h, --help            Show this help message and exit.
    -c CONFIG, --config CONFIG
                            Path to a configuration file in json or yaml format.
    --print_config [={comments,skip_null,skip_default}+]
                            Print configuration and exit.

    subcommands:
    For more details of each subcommand add it as argument followed by --help.

    {fit,validate,test,predict}
        fit                 Runs the full optimization routine.
        validate            Perform one evaluation epoch over the validation set.
        test                Perform one evaluation epoch over the test set.
        predict             Run inference on your data.


The message tells us that we have a few available subcommands:

.. code:: bash

    python main.py [subcommand]

which you can use depending on your use case:

.. code:: bash

    $ python main.py fit
    $ python main.py validate
    $ python main.py test
    $ python main.py predict

----

**************************
Train a model with the CLI
**************************
To train a model, use the ``fit`` subcommand:

.. code:: bash

    python main.py fit

View all available options with the ``--help`` argument given after the subcommand:

.. code:: bash

    $ python main.py fit --help

    usage: main.py [options] fit [-h] [-c CONFIG]
                                [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                                ...
                                [--ckpt_path CKPT_PATH]
        --trainer.logger LOGGER

    optional arguments:
    <class '__main__.DemoModel'>:
        --model.out_dim OUT_DIM
                                (type: int, default: 10)
        --model.learning_rate LEARNING_RATE
                                (type: float, default: 0.02)
    <class 'lightning.pytorch.demos.boring_classes.BoringDataModule'>:
    --data CONFIG         Path to a configuration file.
    --data.data_dir DATA_DIR
                            (type: str, default: ./)

With the Lightning CLI enabled, you can now change the parameters without touching your code:

.. code:: bash

    # change the learning_rate
    python main.py fit --model.learning_rate 0.1

    # change the output dimensions also
    python main.py fit --model.out_dim 10 --model.learning_rate 0.1

    # change trainer and data arguments too
    python main.py fit --model.out_dim 2 --model.learning_rate 0.1 --data.data_dir '~/' --trainer.logger False

.. tip::

    The options that become available in the CLI are the ``__init__`` parameters of the ``LightningModule`` and
    ``LightningDataModule`` classes. Thus, to make hyperparameters configurable, just add them to your class's
    ``__init__``. It is highly recommended that these parameters are described in the docstring so that the CLI shows
    them in the help. Also, the parameters should have accurate type hints so that the CLI can fail early and give
    understandable error messages when incorrect values are given.
