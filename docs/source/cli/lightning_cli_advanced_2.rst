:orphan:

.. testsetup:: *
    :skipif: not _JSONARGPARSE_AVAILABLE

    import torch
    from unittest import mock
    from typing import List
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, Callback


    class NoFitTrainer(Trainer):
        def fit(self, *_, **__):
            pass


    class LightningCLI(pl.utilities.cli.LightningCLI):
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

#######################################
Eliminate config boilerplate (Advanced)
#######################################

******************************
Customize arguments by command
******************************
To customize arguments by subcommand, pass the config *before* the subcommand:

.. code-block:: bash

    $ python main.py [before] [subcommand] [after]
    $ python main.py  ...         fit       ...

For example, here we set the Trainer argument [max_steps = 100] for the full training routine and [max_steps = 10] for testing:

.. code-block:: bash

    # config1.yaml
    fit:
        trainer:
            max_steps: 100
    test:
        trainer:
            max_epochs: 10

now you can toggle this behavior by subcommand:

.. code-block:: bash

    # full routine with max_steps = 100
    $ python main.py --config config1.yaml fit

    # test only with max_epochs = 10
    $ python main.py --config config1.yaml test

----

*********************
Use groups of options
*********************
Groups of options can also be given as independent config files:

.. code-block:: bash

    $ python trainer.py fit --trainer trainer.yaml --model model.yaml --data data.yaml [...]

----

***************************
Run from cloud yaml configs
***************************
For certain enterprise workloads, Lightning CLI supports running from hosted configs:

.. code-block:: bash

    $ python trainer.py [subcommand] --config s3://bucket/config.yaml

For more options, refer to :doc:`Remote filesystems <../common/remote_fs>`.

----

**************************************
Use a config via environment variables
**************************************
For certain CI/CD systems, it's useful to pass in config files as environment variables:

.. code-block:: bash

    $ python trainer.py fit --trainer "$TRAINER_CONFIG" --model "$MODEL_CONFIG" [...]

----

***************************************
Run from environment variables directly
***************************************
The Lightning CLI can convert every possible CLI flag into an environment variable. To enable this, set the *env_parse* argument:

.. code:: python

    LightningCLI(env_parse=True)

now use the ``--help`` CLI flag with any subcommand:

.. code:: bash

    $ python main.py fit --help

which will show you ALL possible environment variables you can now set:

.. code:: bash

    usage: main.py [options] fit [-h] [-c CONFIG]
                                [--trainer.max_epochs MAX_EPOCHS] [--trainer.min_epochs MIN_EPOCHS]
                                [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS]
                                ...
                                [--ckpt_path CKPT_PATH]

    optional arguments:
    ...
    --model CONFIG        Path to a configuration file.
    --model.out_dim OUT_DIM
                            (type: int, default: 10)
    --model.learning_rate LEARNING_RATE
                            (type: float, default: 0.02)

now you can customize the behavior via environment variables:

.. code:: bash

    # set the options via env vars
    $ export LEARNING_RATE=0.01
    $ export OUT_DIM=5

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

For more configuration options, refer to the `ArgumentParser API
<https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_ documentation.

----

*****************************
Enable variable interpolation
*****************************
In certain cases where multiple configs need to share variables, consider using variable interpolation. Variable interpolation
allows you to add variables to your yaml configs like so:

.. code-block:: yaml

    model:
      encoder_layers: 12
      decoder_layers:
      - ${model.encoder_layers}
      - 4

To enable variable interpolation, first install omegaconf:

.. code:: bash

    pip install omegaconf

Once this is installed, the Lightning CLI will automatically handle variables in yaml files:

.. code bash:

    python main.py --model.encoder_layers=12
