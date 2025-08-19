:orphan:

#################################################
Configure hyperparameters from the CLI (Advanced)
#################################################
**Audience:** Users looking to modularize their code for a professional project.

**Pre-reqs:** You must have read :doc:`(Mix models and datasets) <lightning_cli_intermediate_2>`.

As a project becomes more complex, the number of configurable options becomes very large, making it inconvenient to
control through individual command line arguments. To address this, CLIs implemented using
:class:`~lightning.pytorch.cli.LightningCLI` always support receiving input from configuration files. The default format
used for config files is YAML.

.. tip::

    If you are unfamiliar with YAML, it is recommended that you first read :ref:`what-is-a-yaml-config-file`.


----

***********************
Run using a config file
***********************
To run the CLI using a yaml config, do:

.. code:: bash

    python main.py fit --config config.yaml

Individual arguments can be given to override options in the config file:

.. code:: bash

    python main.py fit --config config.yaml --trainer.max_epochs 100

----

************************
Automatic save of config
************************

To ease experiment reporting and reproducibility, by default ``LightningCLI`` automatically saves the full YAML
configuration in the log directory. After multiple fit runs with different hyperparameters, each one will have in its
respective log directory a ``config.yaml`` file. These files can be used to trivially reproduce an experiment, e.g.:

.. code:: bash

    python main.py fit --config lightning_logs/version_7/config.yaml

The automatic saving of the config is done by the special callback :class:`~lightning.pytorch.cli.SaveConfigCallback`.
This callback is automatically added to the ``Trainer``. To disable the save of the config, instantiate ``LightningCLI``
with ``save_config_callback=None``.

.. tip::

    To change the file name of the saved configs to e.g. ``name.yaml``, do:

    .. code:: python

        cli = LightningCLI(..., save_config_kwargs={"config_filename": "name.yaml"})

It is also possible to extend the :class:`~lightning.pytorch.cli.SaveConfigCallback` class, for instance to additionally
save the config in a logger. An example of this is:

    .. code:: python

        class LoggerSaveConfigCallback(SaveConfigCallback):
            def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})


        cli = LightningCLI(..., save_config_callback=LoggerSaveConfigCallback)

.. tip::

    If you want to disable the standard behavior of saving the config to the ``log_dir``, then you can either implement
    ``__init__`` and call ``super().__init__(*args, save_to_log_dir=False, **kwargs)`` or instantiate the
    ``LightningCLI`` as:

    .. code:: python

        cli = LightningCLI(..., save_config_kwargs={"save_to_log_dir": False})

.. note::

    The ``save_config`` method is only called on rank zero. This allows to implement a custom save config without having
    to worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the process
    hang waiting for a broadcast. If you need to make collective calls, implement the ``setup`` method instead.


----

*********************************
Prepare a config file for the CLI
*********************************
The ``--help`` option of the CLIs can be used to learn which configuration options are available and how to use them.
However, writing a config from scratch can be time-consuming and error-prone. To alleviate this, the CLIs have the
``--print_config`` argument, which prints to stdout the configuration without running the command.

For a CLI implemented as ``LightningCLI(DemoModel, BoringDataModule)``, executing:

.. code:: bash

    python main.py fit --print_config

generates a config with all default values like the following:

.. code:: bash

    seed_everything: null
    trainer:
      logger: true
      ...
    model:
      out_dim: 10
      learning_rate: 0.02
    data:
      data_dir: ./
    ckpt_path: null

Other command line arguments can be given and considered in the printed configuration. A use case for this is CLIs that
accept multiple models. By default, no model is selected, meaning the printed config will not include model settings. To
get a config with the default values of a particular model would be:

.. code:: bash

    python main.py fit --model DemoModel --print_config

which generates a config like:

.. code:: bash

    seed_everything: null
    trainer:
      ...
    model:
      class_path: lightning.pytorch.demos.boring_classes.DemoModel
      init_args:
        out_dim: 10
        learning_rate: 0.02
    ckpt_path: null

.. tip::

    A standard procedure to run experiments can be:

    .. code:: bash

        # Print a configuration to have as reference
        python main.py fit --print_config > config.yaml
        # Modify the config to your liking - you can remove all default arguments
        nano config.yaml
        # Fit your model using the edited configuration
        python main.py fit --config config.yaml

Configuration items can be either simple Python objects such as int and str,
or complex objects comprised of a ``class_path`` and ``init_args`` arguments. The ``class_path`` refers
to the complete import path of the item class, while ``init_args`` are the arguments to be passed
to the class constructor. For example, your model is defined as:

.. code:: python

    # model.py
    class MyModel(L.LightningModule):
        def __init__(self, criterion: torch.nn.Module):
            self.criterion = criterion

Then the config would be:

.. code:: yaml

    model:
      class_path: model.MyModel
      init_args:
        criterion:
          class_path: torch.nn.CrossEntropyLoss
          init_args:
            reduction: mean
        ...

``LightningCLI`` uses `jsonargparse <https://github.com/omni-us/jsonargparse>`_ under the hood for parsing
configuration files and automatic creation of objects, so you don't need to do it yourself.

.. note::

    Lightning automatically registers all subclasses of :class:`~lightning.pytorch.core.LightningModule`,
    so the complete import path is not required for them and can be replaced by the class name.

.. note::

    Parsers make a best effort to determine the correct names and types that the parser should accept.
    However, there can be cases not yet supported or cases for which it would be impossible to support.
    To somewhat overcome these limitations, there is a special key ``dict_kwargs`` that can be used
    to provide arguments that will not be validated during parsing, but will be used for class instantiation.

    For example, then using the ``lightning.pytorch.profilers.PyTorchProfiler`` profiler,
    the ``profile_memory`` argument has a type that is determined dynamically. As a result, it's not possible
    to know the expected type during parsing. To account for this, your config file should be set up like this:

    .. code:: yaml

        trainer:
          profiler:
            class_path: lightning.pytorch.profilers.PyTorchProfiler
            dict_kwargs:
              profile_memory: true

----

********************
Compose config files
********************
Multiple config files can be provided, and they will be parsed sequentially. Let's say we have two configs with common
settings:

.. code:: yaml

    # config_1.yaml
    trainer:
      num_epochs: 10
      ...

    # config_2.yaml
    trainer:
      num_epochs: 20
      ...

The value from the last config will be used, ``num_epochs = 20`` in this case:

.. code-block:: bash

    $ python main.py fit --config config_1.yaml --config config_2.yaml

----

*********************
Use groups of options
*********************
Groups of options can also be given as independent config files. For configs like:

.. code:: yaml

    # trainer.yaml
    num_epochs: 10

    # model.yaml
    out_dim: 7

    # data.yaml
    data_dir: ./data

a fit command can be run as:

.. code-block:: bash

    $ python main.py fit --trainer trainer.yaml --model model.yaml --data data.yaml [...]
