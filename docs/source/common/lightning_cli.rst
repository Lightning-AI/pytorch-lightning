.. testsetup:: *
    :skipif: not _JSONARGPARSE_AVAILABLE

    import torch
    from unittest import mock
    from typing import List
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer
    from pytorch_lightning.utilities.cli import LightningCLI

    cli_fit = LightningCLI.fit
    LightningCLI.fit = lambda *_, **__: None
    trainer_fit = Trainer.fit
    Trainer.fit = lambda *_, **__: None


    class MyModel(LightningModule):
        def __init__(
            self,
            encoder_layers: int = 12,
            decoder_layers: List[int] = [2, 4],
            batch_size: int = 8,
        ):
            pass


    class MyClassModel(LightningModule):
        def __init__(self, num_classes: int):
            pass


    class MyDataModule(LightningDataModule):
        def __init__(self, batch_size: int = 8):
            self.num_classes = 5


    def send_email(address, message):
        pass


    MyModelBaseClass = MyModel
    MyDataModuleBaseClass = MyDataModule

    EncoderBaseClass = MyModel
    DecoderBaseClass = MyModel

    mock_argv = mock.patch("sys.argv", ["any.py"])
    mock_argv.start()

.. testcleanup:: *

    LightningCLI.fit = cli_fit
    Trainer.fit = trainer_fit
    mock_argv.stop()


Lightning CLI and config files
------------------------------

Another source of boilerplate code that Lightning can help to reduce is in the implementation of command line tools.
Furthermore, it provides a standardized way to configure trainings using a single file that includes settings for
:class:`~pytorch_lightning.trainer.trainer.Trainer` and user extended
:class:`~pytorch_lightning.core.lightning.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` classes. The full configuration is automatically saved
in the log directory. This has the benefit of greatly simplifying the reproducibility of experiments.

The main requirement for user extended classes to be made configurable is that all relevant init arguments must have
type hints. This is not a very demanding requirement since it is good practice to do anyway. As a bonus if the arguments
are described in the docstrings, then the help of the command line tool will display them.

.. warning:: ``LightningCLI`` is in beta and subject to change.

----------


LightningCLI
^^^^^^^^^^^^

The implementation of training command line tools is done via the :class:`~pytorch_lightning.utilities.cli.LightningCLI`
class. The minimal installation of pytorch-lightning does not include this support. To enable it, either install
lightning with the :code:`all` extras require or install the package :code:`jsonargparse[signatures]`.

The case in which the user's :class:`~pytorch_lightning.core.lightning.LightningModule` class implements all required
:code:`*_dataloader` methods, a :code:`trainer.py` tool can be as simple as:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(MyModel)

The help of the tool describing all configurable options and default values can be shown by running :code:`python
trainer.py --help`. Default options can be changed by providing individual command line arguments. However, it is better
practice to create a configuration file and provide this to the tool. A way to do this would be:

.. code-block:: bash

    # Dump default configuration to have as reference
    python trainer.py --print_config > default_config.yaml
    # Create config including only options to modify
    nano config.yaml
    # Run training using created configuration
    python trainer.py --config config.yaml

The instantiation of the :class:`~pytorch_lightning.utilities.cli.LightningCLI` class takes care of parsing command line
and config file options, instantiating the classes, setting up a callback to save the config in the log directory and
finally running the trainer. The resulting object :code:`cli` can be used for example to get the instance of the model,
(:code:`cli.model`).

After multiple trainings with different configurations, each run will have in its respective log directory a
:code:`config.yaml` file. This file can be used for reference to know in detail all the settings that were used for each
particular run, and also could be used to trivially reproduce a training, e.g.:

.. code-block:: bash

    python trainer.py --config lightning_logs/version_7/config.yaml

If a separate :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class is required, the trainer tool just
needs a small modification as follows:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(MyModel, MyDataModule)

The start of a possible implementation of :class:`MyModel` including the recommended argument descriptions in the
docstring could be the one below. Note that by using type hints and docstrings there is no need to duplicate this
information to define its configurable arguments.

.. testcode:: mymodel

    class MyModel(LightningModule):
        def __init__(self, encoder_layers: int = 12, decoder_layers: List[int] = [2, 4]):
            """Example encoder-decoder model

            Args:
                encoder_layers: Number of layers for the encoder
                decoder_layers: Number of layers for each decoder block
            """
            super().__init__()
            self.save_hyperparameters()

With this model class, the help of the trainer tool would look as follows:

.. code-block:: bash

    $ python trainer.py --help
    usage: trainer.py [-h] [--print_config] [--config CONFIG]
                      [--trainer.logger LOGGER]
                      ...

    pytorch-lightning trainer command line tool

    optional arguments:
      -h, --help            show this help message and exit
      --print_config        print configuration and exit
      --config CONFIG       Path to a configuration file in json or yaml format.
                            (default: null)

    Customize every aspect of training via flags:
      ...
      --trainer.max_epochs MAX_EPOCHS
                            Stop training once this number of epochs is reached.
                            (type: int, default: 1000)
      --trainer.min_epochs MIN_EPOCHS
                            Force training for at least these many epochs (type: int,
                            default: 1)
      ...

    Example encoder-decoder model:
      --model.encoder_layers ENCODER_LAYERS
                            Number of layers for the encoder (type: int, default: 12)
      --model.decoder_layers DECODER_LAYERS
                            Number of layers for each decoder block (type: List[int],
                            default: [2, 4])

The default configuration that option :code:`--print_config` gives is in yaml format and for the example above would
look as follows:

.. code-block:: bash

    $ python trainer.py --print_config
    model:
      decoder_layers:
      - 2
      - 4
      encoder_layers: 12
    trainer:
      accelerator: null
      accumulate_grad_batches: 1
      amp_backend: native
      amp_level: O2
      ...

Note that there is a section for each class (model and trainer) including all the init parameters of the class. This
grouping is also used in the formatting of the help shown previously.


Use of command line arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    $ python trainer.py --config experiment_defaults.yaml --trainer.max_epochs 100

Another common pattern could be having multiple config files:

.. code-block:: bash

    $ python trainer.py --config config1.yaml --config config2.yaml [...]

As explained before, :code:`config1.yaml` is parsed first and then :code:`config2.yaml`. Therefore, if individual
settings are defined in both files, then the ones in :code:`config2.yaml` will be used. Settings in :code:`config1.yaml`
that are not in :code:`config2.yaml` are be kept.

Groups of options can also be given as independent config files:

.. code-block:: bash

    $ python trainer.py --trainer trainer.yaml --model model.yaml --data data.yaml [...]

When running experiments in clusters it could be desired to use a config which needs to be accessed from a remote
location. :class:`~pytorch_lightning.utilities.cli.LightningCLI` comes with `fsspec
<https://filesystem-spec.readthedocs.io/en/stable/>`_ support which allows reading from many types of remote file
systems. One example is if you have installed the `gcsfs <https://gcsfs.readthedocs.io/en/stable/>`_ then a config could
be stored in an S3 bucket and accessed as:

.. code-block:: bash

    $ python trainer.py --config s3://bucket/config.yaml [...]

In some cases people might what to pass an entire config in an environment variable, which could also be used instead of
a path to a file, for example:

.. code-block:: bash

    $ python trainer.py --trainer "$TRAINER_CONFIG" --model "$MODEL_CONFIG" [...]

An alternative for environment variables could be to instantiate the CLI with :code:`env_parse=True`. In this case the
help shows the names of the environment variables for all options. A global config would be given in :code:`PL_CONFIG`
and there wouldn't be a need to specify any command line argument.

It is also possible to set a path to a config file of defaults. If the file exists it would be automatically loaded
without having to specify any command line argument. Arguments given would override the values in the default config
file. Loading a defaults file :code:`my_cli_defaults.yaml` in the current working directory would be implemented as:

.. testcode::

    cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"default_config_files": ["my_cli_defaults.yaml"]})

To load a file in the user's home directory would be just changing to :code:`~/.my_cli_defaults.yaml`. Note that this
setting is given through :code:`parser_kwargs`. More parameters are supported. For details see the `ArgumentParser API
<https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_ documentation.


Instantiation only mode
^^^^^^^^^^^^^^^^^^^^^^^

The CLI is designed to start fitting with minimal code changes. On class instantiation, the CLI will automatically
call ``trainer.fit(...)`` internally so you don't have to do it. To avoid this, you can set the following argument:

.. testcode::

    cli = LightningCLI(MyModel, run=False)  # True by default
    # you'll have to call fit yourself:
    cli.trainer.fit(cli.model)


This can be useful to implement custom logic without having to subclass the CLI, but still using the CLI's instantiation
and argument parsing capabilities.


Trainer Callbacks and arguments with class type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very important argument of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class is the :code:`callbacks`. In
contrast to other more simple arguments which just require numbers or strings, :code:`callbacks` expects a list of
instances of subclasses of :class:`~pytorch_lightning.callbacks.Callback`. To specify this kind of argument in a config
file, each callback must be given as a dictionary including a :code:`class_path` entry with an import path of the class,
and optionally an :code:`init_args` entry with arguments required to instantiate it. Therefore, a simple configuration
file example that defines a couple of callbacks is the following:

.. code-block:: yaml

    trainer:
      callbacks:
        - class_path: pytorch_lightning.callbacks.EarlyStopping
          init_args:
            patience: 5
        - class_path: pytorch_lightning.callbacks.LearningRateMonitor
          init_args:
            ...

Similar to the callbacks, any arguments in :class:`~pytorch_lightning.trainer.trainer.Trainer` and user extended
:class:`~pytorch_lightning.core.lightning.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` classes that have as type hint a class can be configured
the same way using :code:`class_path` and :code:`init_args`.

Lightning optionally simplifies the user command line so that only the :class:`~pytorch_lightning.callbacks.Callback`
name is required. The argument's order matters and the user needs to pass the arguments in the following way.
This is supported only for PyTorch Lightning built-in :class:`~pytorch_lightning.callbacks.Callback`.

.. code-block:: bash

    $ python ... \
        --trainer.callbacks={CALLBACK_1_NAME} \
        --trainer.callbacks.{CALLBACK_1_ARGS_1}=... \
        --trainer.callbacks.{CALLBACK_1_ARGS_2}=... \
        ...
        --trainer.callbacks={CALLBACK_N_NAME} \
        --trainer.callbacks.{CALLBACK_N_ARGS_1}=... \
        ...

Here is an example:

.. code-block:: bash

    $ python ... \
        --trainer.callbacks=EarlyStopping \
        --trainer.callbacks.patience=5 \
        --trainer.callbacks=LearningRateMonitor \
        --trainer.callbacks.logging_interval=epoch

Register your callbacks
^^^^^^^^^^^^^^^^^^^^^^^

Lightning provides registries for you to add your own callbacks and benefit from the command line simplification as described above:

.. code-block:: python

    from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
    from pytorch_lightning.callbacks import Callback


    @CALLBACK_REGISTRY
    class CustomCallback(Callback):
        ...


    cli = LightningCLI(...)

.. code-block:: bash

    $  python ... --trainer.callbacks=CustomCallback ...


Multiple models and/or datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous examples :class:`~pytorch_lightning.utilities.cli.LightningCLI` works only for a single model and
datamodule class. However, there are many cases in which the objective is to easily be able to run many experiments for
multiple models and datasets. For these cases the tool can be configured such that a model and/or a datamodule is
specified by an import path and init arguments. For example, with a tool implemented as:

.. code-block:: python

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(MyModelBaseClass, MyDataModuleBaseClass, subclass_mode_model=True, subclass_mode_data=True)

A possible config file could be as follows:

.. code-block:: yaml

    model:
      class_path: mycode.mymodels.MyModel
      init_args:
        decoder_layers:
        - 2
        - 4
        encoder_layers: 12
    data:
      class_path: mycode.mydatamodules.MyDataModule
      init_args:
        ...
    trainer:
      callbacks:
        - class_path: pytorch_lightning.callbacks.EarlyStopping
          init_args:
            patience: 5
        ...

Only model classes that are a subclass of :code:`MyModelBaseClass` would be allowed, and similarly only subclasses of
:code:`MyDataModuleBaseClass`. If as base classes :class:`~pytorch_lightning.core.lightning.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` are given, then the tool would allow any lightning
module and data module.

.. tip::

    Note that with the subclass modes the :code:`--help` option does not show information for a specific subclass. To
    get help for a subclass the options :code:`--model.help` and :code:`--data.help` can be used, followed by the
    desired class path. Similarly :code:`--print_config` does not include the settings for a particular subclass. To
    include them the class path should be given before the :code:`--print_config` option. Examples for both help and
    print config are:

    .. code-block:: bash

        $ python trainer.py --model.help mycode.mymodels.MyModel
        $ python trainer.py --model mycode.mymodels.MyModel --print_config


Models with multiple submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many use cases require to have several modules each with its own configurable options. One possible way to handle this
with LightningCLI is to implement a single module having as init parameters each of the submodules. Since the init
parameters have as type a class, then in the configuration these would be specified with :code:`class_path` and
:code:`init_args` entries. For instance a model could be implemented as:

.. testcode::

    class MyMainModel(LightningModule):
        def __init__(self, encoder: EncoderBaseClass, decoder: DecoderBaseClass):
            """Example encoder-decoder submodules model

            Args:
                encoder: Instance of a module for encoding
                decoder: Instance of a module for decoding
            """
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

If the CLI is implemented as :code:`LightningCLI(MyMainModel)` the configuration would be as follows:

.. code-block:: yaml

    model:
      encoder:
        class_path: mycode.myencoders.MyEncoder
        init_args:
          ...
      decoder:
        class_path: mycode.mydecoders.MyDecoder
        init_args:
          ...

It is also possible to combine :code:`subclass_mode_model=True` and submodules, thereby having two levels of
:code:`class_path`.


Customizing LightningCLI
^^^^^^^^^^^^^^^^^^^^^^^^

The init parameters of the :class:`~pytorch_lightning.utilities.cli.LightningCLI` class can be used to customize some
things, namely: the description of the tool, enabling parsing of environment variables and additional arguments to
instantiate the trainer and configuration parser.

Nevertheless the init arguments are not enough for many use cases. For this reason the class is designed so that can be
extended to customize different parts of the command line tool. The argument parser class used by
:class:`~pytorch_lightning.utilities.cli.LightningCLI` is
:class:`~pytorch_lightning.utilities.cli.LightningArgumentParser` which is an extension of python's argparse, thus
adding arguments can be done using the :func:`add_argument` method. In contrast to argparse it has additional methods to
add arguments, for example :func:`add_class_arguments` adds all arguments from the init of a class, though requiring
parameters to have type hints. For more details about this please refer to the `respective documentation
<https://jsonargparse.readthedocs.io/en/stable/#classes-methods-and-functions>`_.

The :class:`~pytorch_lightning.utilities.cli.LightningCLI` class has the
:meth:`~pytorch_lightning.utilities.cli.LightningCLI.add_arguments_to_parser` method which can be implemented to include
more arguments. After parsing, the configuration is stored in the :code:`config` attribute of the class instance. The
:class:`~pytorch_lightning.utilities.cli.LightningCLI` class also has two methods that can be used to run code before
and after :code:`trainer.fit` is executed: :meth:`~pytorch_lightning.utilities.cli.LightningCLI.before_fit` and
:meth:`~pytorch_lightning.utilities.cli.LightningCLI.after_fit`. A realistic example for these would be to send an email
before and after the execution of fit. The code would be something like:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_argument("--notification_email", default="will@email.com")

        def before_fit(self):
            send_email(address=self.config["notification_email"], message="trainer.fit starting")

        def after_fit(self):
            send_email(address=self.config["notification_email"], message="trainer.fit finished")


    cli = MyLightningCLI(MyModel)

Note that the config object :code:`self.config` is a dictionary whose keys are global options or groups of options. It
has the same structure as the yaml format described previously. This means for instance that the parameters used for
instantiating the trainer class can be found in :code:`self.config['trainer']`.

.. tip::

    Have a look at the :class:`~pytorch_lightning.utilities.cli.LightningCLI` class API reference to learn about other
    methods that can be extended to customize a CLI.


Configurable callbacks
^^^^^^^^^^^^^^^^^^^^^^

As explained previously, any callback can be added by including it in the config via :code:`class_path` and
:code:`init_args` entries. However, there are other cases in which a callback should always be present and be
configurable. This can be implemented as follows:

.. testcode::

    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.utilities.cli import LightningCLI


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_lightning_class_args(EarlyStopping, "my_early_stopping")
            parser.set_defaults({"my_early_stopping.patience": 5})


    cli = MyLightningCLI(MyModel)

To change the configuration of the :code:`EarlyStopping` in the config it would be:

.. code-block:: yaml

    model:
      ...
    trainer:
      ...
    my_early_stopping:
      patience: 5

.. note::

    The example above overrides a default in :code:`add_arguments_to_parser`. This is included to show that defaults can
    be changed if needed. However, note that overriding of defaults in the source code is not intended to be used to
    store the best hyperparameters for a task after experimentation. To ease reproducibility the source code should be
    stable. It is better practice to store the best hyperparameters for a task in a configuration file independent from
    the source code.


Class type defaults
^^^^^^^^^^^^^^^^^^^

The support for classes as type hints allows to try many possibilities with the same CLI. This is a useful feature, but
it can make it tempting to use an instance of a class as a default. For example:

.. testcode::

    class MyMainModel(LightningModule):
        def __init__(
            self,
            backbone: torch.nn.Module = MyModel(encoder_layers=24),  # BAD PRACTICE!
        ):
            super().__init__()
            self.backbone = backbone

Normally classes are mutable as it is in this case. The instance of :code:`MyModel` would be created the moment that the
module that defines :code:`MyMainModel` is first imported. This means that the default of :code:`backbone` will be
initialized before the CLI class runs :code:`seed_everything` making it non-reproducible. Furthermore, if
:code:`MyMainModel` is used more than once in the same Python process and the :code:`backbone` parameter is not
overridden, the same instance would be used in multiple places which very likely is not what the developer intended.
Having an instance as default also makes it impossible to generate the complete config file since for arbitrary classes
it is not known which arguments were used to instantiate it.

A good solution to these problems is to not have a default or set the default to a special value (e.g. a
string) which would be checked in the init and instantiated accordingly. If a class parameter has no default and the CLI
is subclassed then a default can be set as follows:

.. testcode::

    default_backbone = {
        "class_path": "import.path.of.MyModel",
        "init_args": {
            "encoder_layers": 24,
        },
    }


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"model.backbone": default_backbone})

A more compact version that avoids writing a dictionary would be:

.. testcode::

    from jsonargparse import lazy_instance


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"model.backbone": lazy_instance(MyModel, encoder_layers=24)})


Argument linking
^^^^^^^^^^^^^^^^

Another case in which it might be desired to extend :class:`~pytorch_lightning.utilities.cli.LightningCLI` is that the
model and data module depend on a common parameter. For example in some cases both classes require to know the
:code:`batch_size`. It is a burden and error prone giving the same value twice in a config file. To avoid this the
parser can be configured so that a value is only given once and then propagated accordingly. With a tool implemented
like shown below, the :code:`batch_size` only has to be provided in the :code:`data` section of the config.

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size")


    cli = MyLightningCLI(MyModel, MyDataModule)

The linking of arguments is observed in the help of the tool, which for this example would look like:

.. code-block:: bash

    $ python trainer.py --help
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

    from pytorch_lightning.utilities.cli import LightningCLI


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")


    cli = MyLightningCLI(MyClassModel, MyDataModule)

Instantiation links are used to automatically determine the order of instantiation, in this case data first.

.. tip::

    The linking of arguments can be used for more complex cases. For example to derive a value via a function that takes
    multiple settings as input. For more details have a look at the API of `link_arguments
    <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.link_arguments>`_.


Optimizers and learning rate schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimizers and learning rate schedulers can also be made configurable. The most common case is when a model only has a
single optimizer and optionally a single learning rate scheduler. In this case the model's
:class:`~pytorch_lightning.core.lightning.LightningModule` could be left without implementing the
:code:`configure_optimizers` method since it is normally always the same and just adds boilerplate. The following code
snippet shows how to implement it:

.. testcode::

    import torch
    from pytorch_lightning.utilities.cli import LightningCLI


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)


    cli = MyLightningCLI(MyModel)

With this the :code:`configure_optimizers` method is automatically implemented and in the config the :code:`optimizer`
and :code:`lr_scheduler` groups would accept all of the options for the given classes, in this example :code:`Adam` and
:code:`ExponentialLR`. Therefore, the config file would be structured like:

.. code-block:: yaml

    optimizer:
      lr: 0.01
    lr_scheduler:
      gamma: 0.2
    model:
      ...
    trainer:
      ...

And any of these arguments could be passed directly through command line. For example:

.. code-block:: bash

    $ python train.py --optimizer.lr=0.01 --lr_scheduler.gamma=0.2

There is also the possibility of selecting among multiple classes by giving them as a tuple. For example:

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))

In this case in the config the :code:`optimizer` group instead of having directly init settings, it should specify
:code:`class_path` and optionally :code:`init_args`. Sub-classes of the classes in the tuple would also be accepted.
A corresponding example of the config file would be:

.. code-block:: yaml

    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: 0.01

And the same through command line:

.. code-block:: bash

    $ python train.py --optimizer.class_path=torch.optim.Adam --optimizer.init_args.lr=0.01

Optionally, the command line can be simplified for PyTorch built-in `optimizers` and `schedulers`:

.. code-block:: bash

    $ python train.py --optimizer=Adam --optimizer.lr=0.01

The automatic implementation of :code:`configure_optimizers` can be disabled by linking the configuration group. An
example can be :code:`ReduceLROnPlateau` which requires to specify a monitor. This would be:

.. testcode::

    from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI


    class MyModel(LightningModule):
        def __init__(self, optimizer_init: dict, lr_scheduler_init: dict):
            super().__init__()
            self.optimizer_init = optimizer_init
            self.lr_scheduler_init = lr_scheduler_init

        def configure_optimizers(self):
            optimizer = instantiate_class(self.parameters(), self.optimizer_init)
            scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(
                torch.optim.Adam,
                link_to="model.optimizer_init",
            )
            parser.add_lr_scheduler_args(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                link_to="model.lr_scheduler_init",
            )


    cli = MyLightningCLI(MyModel)

For both possibilities of using :meth:`pytorch_lightning.utilities.cli.LightningArgumentParser.add_optimizer_args` with
a single class or a tuple of classes, the value given to :code:`optimizer_init` will always be a dictionary including
:code:`class_path` and :code:`init_args` entries. The function
:func:`~pytorch_lightning.utilities.cli.instantiate_class` takes care of importing the class defined in
:code:`class_path` and instantiating it using some positional arguments, in this case :code:`self.parameters()`, and the
:code:`init_args`. Any number of optimizers and learning rate schedulers can be added when using :code:`link_to`.

Built in schedulers & optimizers and registering your own
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For code simplification, the CLI provides properties with PyTorch's built-in optimizers and learning rate schedulers
already registered.
Only the optimizer or scheduler name needs to be passed along its arguments.

.. code-block:: bash

    $ python train.py --optimizer=Adam --optimizer.lr=0.01 --lr_scheduler=CosineAnnealingLR

If your model requires multiple optimizers, you can choose from all available optimizers and learning rate schedulers
by accessing `self.registered_optimizers` and `self.registered_lr_schedulers` respectively.

.. code-block::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(
                self.registered_optimizers,
                nested_key="gen_optimizer",
                link_to="model.optimizer_init",
            )
            parser.add_optimizer_args(
                self.registered_optimizers,
                nested_key="gen_discriminator",
                link_to="model.optimizer_init",
            )

.. code-block:: bash

    $ python train.py --gen_optimizer=Adam --optimizer.lr=0.01 --gen_discriminator=Adam --optimizer.lr=0.0001

Furthermore, you can register your own optimizers and/or learning rate schedulers as follows:

.. code-block:: python

    import torch
    from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY
    from pytorch_lightning.callbacks import Callback


    @OPTIMIZER_REGISTRY
    class CustomAdam(torch.optim.Adam):
        ...


    @LR_SCHEDULER_REGISTRY
    class CustomCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
        ...


    cli = LightningCLI(...)

.. code-block:: bash

    $ python train.py --optimizer=CustomAdam --optimizer.lr=0.01 --lr_scheduler=CustomCosineAnnealingLR


Notes related to reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The topic of reproducibility is complex and it is impossible to guarantee reproducibility by just providing a class that
people can use in unexpected ways. Nevertheless :class:`~pytorch_lightning.utilities.cli.LightningCLI` tries to give a
framework and recommendations to make reproducibility simpler.

When an experiment is run, it is good practice to use a stable version of the source code, either being a released
package or at least a commit of some version controlled repository. For each run of a CLI the config file is
automatically saved including all settings. This is useful to figure out what was done for a particular run without
requiring to look at the source code. If by mistake the exact version of the source code is lost or some defaults
changed, having the full config means that most of the information is preserved.

The class is targeted at implementing CLIs because running a command from a shell provides a separation with the Python
source code. Ideally the CLI would be placed in your path as part of the installation of a stable package, instead of
running from a clone of a repository that could have uncommitted local modifications. Creating installable packages that
include CLIs is out of the scope of this document. This is mentioned only as a teaser for people who would strive for
the best practices possible.
