Trainer CLI and config files
----------------------------

Another source of boilerplate code that Lightning can help to reduce is in the
implementation of training command line tools. Furthermore, it provides a
standardized way to configure trainings using a single file that includes
settings for :class:`~pytorch_lightning.trainer.trainer.Trainer` and user
extended :class:`~pytorch_lightning.core.lightning.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` classes. The
full configuration is automatically saved in the log directory. This has the
benefit of greatly simplifying the reproducibility of experiments.

The main requirement for user extended classes to be made configurable is that
all relevant init arguments must have type hints. This is not a very demanding
requirement since anyway it is good practice to do anyway. As a bonus if the
arguments are also described in the docstrings, then the help of the training
tool will display them.

----------


training_cli
^^^^^^^^^^^^

The case in which the user's :class:`LightningModule` class implements all
required :code:`*_dataloader` methods, a :code:`trainer.py` tool can be as
simple as:

.. code-block:: python

    from pytorch_lightning.utilities.jsonargparse_utils import trainer_cli
    from mycode import LitModel

    trainer_cli(LitModel)

The help of the tool describing all configurable options and default values can
be shown by running :code:`python trainer.py --help`. Default options can be
changed by providing individual command line arguments. However, it is better
practice to create a configuration file and provide this to the trainer. A way
to do this would be:

.. code-block:: bash

    # Dump default configuration to have as reference
    python trainer.py --print-config > default_config.yaml
    # Create config including only options to modify
    nano config.yaml
    # Run training using created configuration
    python trainer.py --cfg config.yaml

The call to the :func:`trainer_cli` function takes care of parsing command line
and config file options, instantiating the classes, setting up a callback to
save the config in the log directory and finally running :func:`trainer.fit`.

After multiple trainings with different configurations, a previous run can be
trivially reproduced by using the config in the respective log directory, e.g.:

.. code-block:: bash

    python trainer.py --cfg lightning_logs/version_7/config.yaml

The start of a possible implementation of :class:`LitModel` including the
recommended argument descriptions in the docstring could be the one below. Note
that by using type hints and docstrings there is no need to duplicate this
information to define its configurable arguments.

.. code-block:: python

    class LitModel(LightningModule):

        def __init__(self,
                     encoder_layers: int = 12,
                     decoder_layers: List[int] = [2, 4]):
            """Example encoder-decoder model

            Args:
                encoder_layers: Number of layers for the encoder
                decoder_layers: Number of layers for each decoder block
            """
            ...

If a separate :class:`LightningDataModule` class is required, the trainer tool
just needs a small modification as follows:

.. code-block:: python

    from pytorch_lightning.utilities.jsonargparse_utils import trainer_cli
    from mycode import LitModel, LitDataModule

    trainer_cli(LitModel, LitDataModule)


LightningArgumentParser
^^^^^^^^^^^^^^^^^^^^^^^

Even though :func:`trainer_cli` can reduce boilerplate code to a minimum,
clearly there are cases in which it is not enough. For this Lightning provides
the :class:`LightningArgumentParser` class which is an extension of the built-in
Python ArgumentParser that makes it very simple to implement configurable
training tools with the same features as :func:`trainer_cli`.

An example of a more complex training tool could be one in which there are
several independent modules that require configuration. The code for such a case
could look something like:

.. code-block:: python

    from pytorch_lightning.utilities.jsonargparse_utils import LightningArgumentParser, SaveConfigCallback
    from mycode import LitModule1, LitModule2, LitModel, LitDataModule

    # Define parser
    parser = LightningArgumentParser(description='pytorch-lightning trainer',
                                     parse_as_dict=True)
    parser.add_trainer_args()
    parser.add_module_args(LitModule1, 'module1')
    parser.add_module_args(LitModule2, 'module2')
    parser.add_datamodule_args(LitDataModule)

    # Parse configuration
    config = parser.parse_args()

    # Instantiate classes
    module1 = LitModule1(**config['module1'])
    module2 = LitModule2(**config['module2'])
    model = LitModel(module1, module2)
    datamodule = LitDataModule(**config['data'])
    config['trainer']['callbacks'] = [SaveConfigCallback(parser, config)]
    trainer = Trainer(**config['trainer'])

    # Start training
    trainer.fit(model, datamodule)

Note that the configuration object has all options for each module, data and
trainer in different dict keys. The structure of the yaml configuration file is
analogous. Reproducing the training can also be done with the config saved in
the log directory.

The parser is like any other from argparse, thus it can be used to include
global options, for example:

.. code-block:: python

    parser.add_argument('--notification_email', default='will@email.com')

The argument parser is also able to parse environment variables. To enable this
feature, initialize :class:`LightningArgumentParser` including
:code:`default_env=True, env_prefix='PL'`. With this for instance the
:code:`PL_TRAINER__MAX_EPOCHS` environment variable if set would be used to
override the default :code:`max_epochs` of the trainer. Similarly options for
the data module could be set using variables that start with :code:`PL_DATA_`
and likewise for the modules.

Arguments from any other class that have appropriate type hints can also be
added. An example which would store the options for a class :class:`MyClass` in
the :code:`myclass` key of the configuration object would be
:code:`parser.add_class_arguments(MyClass, 'myclass')`.
