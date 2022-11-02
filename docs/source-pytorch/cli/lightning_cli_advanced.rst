:orphan:

#################################################
Configure hyperparameters from the CLI (Advanced)
#################################################
**Audience:** Users looking to modularize their code for a professional project.

**Pre-reqs:** You must have read :doc:`(Mix models and datasets) <lightning_cli_intermediate_2>`.

As a project becomes more complex, the number of configurable options becomes very large, making it inconvenient to
control through individual command line arguments. To address this, CLIs implemented using
:class:`~pytorch_lightning.cli.LightningCLI` always support receiving input from configuration files. The default format
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

The automatic saving of the config is done by the special callback :class:`~pytorch_lightning.cli.SaveConfigCallback`.
This callback is automatically added to the ``Trainer``. To disable the save of the config, instantiate ``LightningCLI``
with ``save_config_callback=None``.

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
      class_path: pytorch_lightning.demos.boring_classes.DemoModel
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
