:orphan:

.. testsetup:: *
    :skipif: not _JSONARGPARSE_AVAILABLE

    import torch
    from unittest import mock
    from typing import List
    import lightning.pytorch.cli as pl_cli
    from lightning.pytorch import LightningModule, LightningDataModule, Trainer, Callback


    class NoFitTrainer(Trainer):
        def fit(self, *_, **__):
            pass


    class LightningCLI(pl_cli.LightningCLI):
        def __init__(self, *args, trainer_class=NoFitTrainer, run=False, **kwargs):
            super().__init__(*args, trainer_class=trainer_class, run=run, **kwargs)


    class MyModel(LightningModule):
        def __init__(
            self,
            encoder_layers: int = 12,
            decoder_layers: List[int] = [2, 4],
            batch_size: int = 8,
        ):
            pass


    class MyDataModule(LightningDataModule):
        def __init__(self, batch_size: int = 8):
            self.num_classes = 5


    mock_argv = mock.patch("sys.argv", ["any.py"])
    mock_argv.start()

.. testcleanup:: *

    mock_argv.stop()

#################################################
Configure hyperparameters from the CLI (Advanced)
#################################################

*********************************
Customize arguments by subcommand
*********************************
To customize arguments by subcommand, pass the config *before* the subcommand:

.. code-block:: bash

    $ python main.py [before] [subcommand] [after]
    $ python main.py  ...         fit       ...

For example, here we set the Trainer argument [max_steps = 100] for the full training routine and [max_steps = 10] for
testing:

.. code-block:: bash

    # config.yaml
    fit:
        trainer:
            max_steps: 100
    test:
        trainer:
            max_epochs: 10

now you can toggle this behavior by subcommand:

.. code-block:: bash

    # full routine with max_steps = 100
    $ python main.py --config config.yaml fit

    # test only with max_epochs = 10
    $ python main.py --config config.yaml test

----

***************************
Run from cloud yaml configs
***************************
For certain enterprise workloads, Lightning CLI supports running from hosted configs:

.. code-block:: bash

    $ python main.py [subcommand] --config s3://bucket/config.yaml

For more options, refer to :doc:`Remote filesystems <../common/remote_fs>`.

----

**************************************
Use a config via environment variables
**************************************
For certain CI/CD systems, it's useful to pass in raw yaml config as environment variables:

.. code-block:: bash

    $ python main.py fit --trainer "$TRAINER_CONFIG" --model "$MODEL_CONFIG" [...]

----

***************************************
Run from environment variables directly
***************************************
The Lightning CLI can convert every possible CLI flag into an environment variable. To enable this, add to
``parser_kwargs`` the ``default_env`` argument:

.. code:: python

    cli = LightningCLI(..., parser_kwargs={"default_env": True})

now use the ``--help`` CLI flag with any subcommand:

.. code:: bash

    $ python main.py fit --help

which will show you ALL possible environment variables that can be set:

.. code:: bash

    usage: main.py [options] fit [-h] [-c CONFIG]
                                ...

    optional arguments:
    ...
    ARG:   --model.out_dim OUT_DIM
    ENV:   PL_FIT__MODEL__OUT_DIM
                            (type: int, default: 10)
    ARG:   --model.learning_rate LEARNING_RATE
    ENV:   PL_FIT__MODEL__LEARNING_RATE
                            (type: float, default: 0.02)

now you can customize the behavior via environment variables:

.. code:: bash

    # set the options via env vars
    $ export PL_FIT__MODEL__LEARNING_RATE=0.01
    $ export PL_FIT__MODEL__OUT_DIM=5

    $ python main.py fit

----

************************
Set default config files
************************
To set a path to a config file of defaults, use the ``default_config_files`` argument:

.. testcode::

    cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"default_config_files": ["my_cli_defaults.yaml"]})

or if you want defaults per subcommand:

.. testcode::

    cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"fit": {"default_config_files": ["my_fit_defaults.yaml"]}})

----

*****************************
Enable variable interpolation
*****************************
In certain cases where multiple settings need to share a value, consider using variable interpolation. For instance:

.. code-block:: yaml

    model:
      encoder_layers: 12
      decoder_layers:
      - ${model.encoder_layers}
      - 4

To enable variable interpolation, first install omegaconf:

.. code:: bash

    pip install omegaconf

Then set omegaconf when instantiating the ``LightningCLI`` class:

.. code:: python

    cli = LightningCLI(MyModel, parser_kwargs={"parser_mode": "omegaconf"})

After this, the CLI will automatically perform interpolation in yaml files:

.. code:: bash

    python main.py --model.encoder_layers=12

For more details about the interpolation support and its limitations, have a look at the `jsonargparse
<https://jsonargparse.readthedocs.io/en/stable/#variable-interpolation>`__ and the `omegaconf
<https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation>`__ documentations.

.. note::

    There are many use cases in which variable interpolation is not the correct approach. When a parameter **must
    always** be derived from other settings, it shouldn't be up to the CLI user to do this in a config file. For
    example, if the data and model both require ``batch_size`` and must be the same value, then
    :ref:`cli_link_arguments` should be used instead of interpolation.
