:orphan:

###########################################
Eliminate config boilerplate (Intermediate)
###########################################
**Audience:** Users who want advanced modularity via the commandline interface (CLI).

**Pre-reqs:** You must already understand how to use a commandline and :doc:`LightningDataModule <../data/datamodule>`.

----

***************************
What is config boilerplate?
***************************
As Lightning projects grow in complexity it becomes desirable to enable full customizability from the commandline (CLI) so you can
change any hyperparameters without changing your code:

.. code:: bash

    # Mix and match anything
    $ python main.py fit --model.learning_rate 0.02
    $ python main.py fit --model.learning_rate 0.01 --trainer.fast_dev_run True

This is what the Lightning CLI enables. Without the Lightning CLI, you usually end up with a TON of boilerplate that looks like this:

.. code:: python

    from argparse import ArgumentParser

    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--learning_rate_1", default=0.02)
        parser.add_argument("--learning_rate_2", default=0.03)
        parser.add_argument("--model", default="cnn")
        parser.add_argument("--command", default="fit")
        parser.add_argument("--run_fast", default=True)
        ...
        # add 100 more of these
        ...

        args = parser.parse_args()

        if args.model == "cnn":
            model = ConvNet(learning_rate=args.learning_rate_1)
        elif args.model == "transformer":
            model = Transformer(learning_rate=args.learning_rate_2)
        trainer = Trainer(fast_dev_run=args.run_fast)
        ...

        if args.command == "fit":
            trainer.fit()
        elif args.command == "test":
            ...

This kind of boilerplate is unsustainable as projects grow in complexity.

----

************************
Enable the Lightning CLI
************************
To enable the Lightning CLI install the extras:

.. code:: bash

    pip install pytorch-lightning[extra]

if the above fails, only install jsonargparse:

.. code:: bash

    pip install -U jsonargparse[signatures]

----

**************************
Connect a model to the CLI
**************************
The simplest way to control a model with the CLI is to wrap it in the LightningCLI object:

.. code:: python

    # main.py
    import torch
    from pytorch_lightning.utilities.cli import LightningCLI

    # simple demo classes for your convenience
    from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!

Now your model can be managed via the CLI. To see the available commands type:

.. code:: bash

    $ python main.py --help

Which prints out:

.. code:: bash

    usage: a.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
            {fit,validate,test,predict,tune} ...

    pytorch-lightning trainer command line tool

    optional arguments:
    -h, --help            Show this help message and exit.
    -c CONFIG, --config CONFIG
                            Path to a configuration file in json or yaml format.
    --print_config [={comments,skip_null,skip_default}+]
                            Print configuration and exit.

    subcommands:
    For more details of each subcommand add it as argument followed by --help.

    {fit,validate,test,predict,tune}
        fit                 Runs the full optimization routine.
        validate            Perform one evaluation epoch over the validation set.
        test                Perform one evaluation epoch over the test set.
        predict             Run inference on your data.
        tune                Runs routines to tune hyperparameters before training.


the message tells us that we have a few available subcommands:

.. code:: bash

    python main.py [subcommand]

which you can use depending on your use case:

.. code:: bash

    $ python main.py fit
    $ python main.py validate
    $ python main.py test
    $ python main.py predict
    $ python main.py tune

----

**************************
Train a model with the CLI
**************************
To run the full training routine (train, val, test), use the subcommand ``fit``:

.. code:: bash

    python main.py fit

View all available options with the ``--help`` command:

.. code:: bash

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
    <class 'pytorch_lightning.demos.boring_classes.BoringDataModule'>:
    --data CONFIG         Path to a configuration file.
    --data.data_dir DATA_DIR
                            (type: str, default: ./)

With the Lightning CLI enabled, you can now change the parameters without touching your code:

.. code:: bash

    # change the learning_rate
    python main.py fit --model.out_dim 30

    # change the out dimensions also
    python main.py fit --model.out_dim 10 --model.learning_rate 0.1

    # change trainer and data arguments too
    python main.py fit --model.out_dim 2 --model.learning_rate 0.1 --data.data_dir '~/' --trainer.logger False
