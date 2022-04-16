#######################################
Eliminate config boilerplate (Advanced)
#######################################


********************
What is a yaml file?
********************

----

**********************************************
Automatically write a config yaml from the CLI
**********************************************
If you would like to have a copy of the configuration that produced this model, you can save a *yaml* file from the *--print_config* outputs:

.. code:: bash

    python main.py fit --model.learning_rate 0.001 --print_config > config.yaml 


You can then use that config to run the same exact model at a later time:

.. code:: bash

    python main.py fit --config config.yaml

----

******************
Compose yaml files
******************

For every CLI implemented, users are encouraged to learn how to run it by reading the documentation printed with the
:code:`--help` option and use the :code:`--print_config` option to guide the writing of config files. A few more details
that might not be clear by only reading the help are the following.

:class:`~pytorch_lightning.utilities.cli.LightningCLI` is based on argparse and as such follows the same arguments style
as many POSIX command line tools. Long options are prefixed with two dashes and its corresponding values should be
provided with an empty space or an equal sign, as :code:`--option value` or :code:`--option=value`. Command line options
are parsed from left to right, therefore if a setting appears multiple times the value most to the right will override
the previous ones. If a class has an init parameter that is required (i.e. no default value), it is given as
:code:`--option` which makes it explicit and more readable instead of relying on positional arguments.

When calling a CLI, all options can be provided using individual arguments. However, given the large amount of options
that the CLIs have, it is recommended to use a combination of config files and individual arguments. Therefore, a common
pattern could be a single config file and only a few individual arguments that override defaults or values in the
config, for example:

.. code-block:: bash

    $ python trainer.py fit --config experiment_defaults.yaml --trainer.max_epochs 100

Another common pattern could be having multiple config files:

.. code-block:: bash

    $ python trainer.py --config config1.yaml --config config2.yaml test --config config3.yaml [...]

As explained before, :code:`config1.yaml` is parsed first and then :code:`config2.yaml`. Therefore, if individual
settings are defined in both files, then the ones in :code:`config2.yaml` will be used. Settings in :code:`config1.yaml`
that are not in :code:`config2.yaml` are be kept. The same happens for :code:`config3.yaml`.

The configuration files before the subcommand (``test`` in this case) can contain custom configuration for multiple of
them, for example:

.. code-block:: bash

    $ cat config1.yaml
    fit:
        trainer:
            limit_train_batches: 100
            max_epochs: 10
    test:
        trainer:
            limit_test_batches: 10


whereas the configuration files passed after the subcommand would be:

.. code-block:: bash

    $ cat config3.yaml
    trainer:
        limit_train_batches: 100
        max_epochs: 10
    # the argument passed to `trainer.test(ckpt_path=...)`
    ckpt_path: "a/path/to/a/checkpoint"


Groups of options can also be given as independent config files:

.. code-block:: bash

    $ python trainer.py fit --trainer trainer.yaml --model model.yaml --data data.yaml [...]

*************************
Use environment variables
*************************

When running experiments in clusters it could be desired to use a config which needs to be accessed from a remote
location. :class:`~pytorch_lightning.utilities.cli.LightningCLI` comes with `fsspec
<https://filesystem-spec.readthedocs.io/en/stable/>`_ support which allows reading and writing from many types of remote
file systems. One example is if you have installed `s3fs <https://s3fs.readthedocs.io/en/latest/>`_ then a config
could be stored in an S3 bucket and accessed as:

.. code-block:: bash

    $ python trainer.py --config s3://bucket/config.yaml [...]

In some cases people might what to pass an entire config in an environment variable, which could also be used instead of
a path to a file, for example:

.. code-block:: bash

    $ python trainer.py fit --trainer "$TRAINER_CONFIG" --model "$MODEL_CONFIG" [...]

An alternative for environment variables could be to instantiate the CLI with :code:`env_parse=True`. In this case the
help shows the names of the environment variables for all options. A global config would be given in :code:`PL_CONFIG`
and there wouldn't be a need to specify any command line argument.

It is also possible to set a path to a config file of defaults. If the file exists it would be automatically loaded
without having to specify any command line argument. Arguments given would override the values in the default config
file. Loading a defaults file :code:`my_cli_defaults.yaml` in the current working directory would be implemented as:

.. testcode::

    cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"default_config_files": ["my_cli_defaults.yaml"]})

or if you want defaults per subcommand:

.. testcode::

    cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"fit": {"default_config_files": ["my_fit_defaults.yaml"]}})

To load a file in the user's home directory would be just changing to :code:`~/.my_cli_defaults.yaml`. Note that this
setting is given through :code:`parser_kwargs`. More parameters are supported. For details see the `ArgumentParser API
<https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_ documentation.
