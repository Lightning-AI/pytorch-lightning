#######################################
Eliminate config boilerplate (Advanced)
#######################################


***************************
What is a yaml config file?
***************************
A yaml is a standard configuration file that describes parameters for sections of a program. It is a common tool in engineering, and it has recently started to gain popularity in machine learning.

.. code:: yaml

    # file.yaml
    car:
        max_speed:100
        max_passengers:2
    plane:
        fuel_capacity: 50
    class_3:
        option_1: 'x'
        option_2: 'y'

----

********************************
Write a config yaml from the CLI
********************************
To have a copy of the configuration that produced this model, save a *yaml* file from the *--print_config* outputs:

.. code:: bash

    python main.py fit --model.learning_rate 0.001 --print_config > config.yaml 

----

**********************
Run from a single yaml
**********************
To run from a yaml, pass a yaml produced with ``--print_config`` to the ``--config`` argument:

.. code:: bash

    python main.py fit --config config.yaml

when using a yaml to run, you can still pass in inline arguments

.. code:: bash

    python main.py fit --config config.yaml --trainer.max_epochs 100

----

******************
Compose yaml files
******************
For production or complex research projects it's advisable to have each object in its own config file. To compose all the configs, pass them all inline:

.. code-block:: bash

    $ python trainer.py --config trainer.yaml --config datamodules.yaml test --config models.yaml ...

The configs will be parsed sequentially. Let's say we have two configs with the same args:

.. code:: yaml

    # trainer_1.yaml
    trainer:
        num_epochs: 10 
    

    # trainer_2.yaml
    trainer:
        num_epochs: 20 

the ones from the last config will be used (num_epochs = 20) in this case:

.. code-block:: bash

    $ python trainer.py --config trainer_2.yaml --config trainer_2.yaml


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

----

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

----

************************
Connect two config files
************************


Argument linking
^^^^^^^^^^^^^^^^

Another case in which it might be desired to extend :class:`~pytorch_lightning.utilities.cli.LightningCLI` is that the
model and data module depend on a common parameter. For example in some cases both classes require to know the
:code:`batch_size`. It is a burden and error prone giving the same value twice in a config file. To avoid this the
parser can be configured so that a value is only given once and then propagated accordingly. With a tool implemented
like shown below, the :code:`batch_size` only has to be provided in the :code:`data` section of the config.

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size")


    cli = MyLightningCLI(MyModel, MyDataModule)

The linking of arguments is observed in the help of the tool, which for this example would look like:

.. code-block:: bash

    $ python trainer.py fit --help
      ...
        --data.batch_size BATCH_SIZE
                              Number of samples in a batch (type: int, default: 8)

      Linked arguments:
        model.batch_size <-- data.batch_size
                              Number of samples in a batch (type: int)

Sometimes a parameter value is only available after class instantiation. An example could be that your model requires
the number of classes to instantiate its fully connected layer (for a classification task) but the value is not
available until the data module has been instantiated. The code below illustrates how to address this.

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")


    cli = MyLightningCLI(MyClassModel, MyDataModule)

Instantiation links are used to automatically determine the order of instantiation, in this case data first.

.. tip::

    The linking of arguments can be used for more complex cases. For example to derive a value via a function that takes
    multiple settings as input. For more details have a look at the API of `link_arguments
    <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.link_arguments>`_.


Variable Interpolation
^^^^^^^^^^^^^^^^^^^^^^

The linking of arguments is intended for things that are meant to be non-configurable. This improves the CLI user
experience since it avoids the need for providing more parameters. A related concept is
variable interpolation which in contrast keeps things being configurable.

The YAML standard defines anchors and aliases which is a way to reuse the content in multiple places of the YAML. This is
supported in the ``LightningCLI`` though it has limitations. Support for OmegaConf's more powerful `variable
interpolation <https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation>`__ will be available
out of the box if this package is installed. To install it run :code:`pip install omegaconf`. Then to enable the use
of OmegaConf in a ``LightningCLI``, when instantiating a parameter needs to be given for the parser as follows:

.. testcode::

    cli = LightningCLI(MyModel, parser_kwargs={"parser_mode": "omegaconf"})

With the encoder-decoder example model above a possible YAML that uses variable interpolation could be the following:

.. code-block:: yaml

    model:
      encoder_layers: 12
      decoder_layers:
      - ${model.encoder_layers}
      - 4

