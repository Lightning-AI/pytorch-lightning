.. testsetup:: *

    from typing import List
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.core.datamodule import LightningDataModule
    from pytorch_lightning.utilities.cli import LightningCLI

    original_fit = LightningCLI.fit
    LightningCLI.fit = lambda self: None

    class MyModel(LightningModule):
        def __init__(
            self,
            encoder_layers: int = 12,
            decoder_layers: List[int] = [2, 4]
        ):
            """Example encoder-decoder model

            Args:
                encoder_layers: Number of layers for the encoder
                decoder_layers: Number of layers for each decoder block
            """
            pass

    class MyDataModule(LightningDataModule):
        pass

    def send_email(address, message):
        pass

    MyModelBaseClass = MyModel
    MyDataModuleBaseClass = MyDataModule

.. testcleanup:: *

    LightningCLI.fit = original_fit


Lightning CLI and config files
------------------------------

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
requirement since it is good practice to do anyway. As a bonus if the arguments
are described in the docstrings, then the help of the training tool will display
them.

----------


LightningCLI
^^^^^^^^^^^^

The case in which the user's
:class:`~pytorch_lightning.core.lightning.LightningModule` class implements all
required :code:`*_dataloader` methods, a :code:`trainer.py` tool can be as
simple as:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(MyModel)

The help of the tool describing all configurable options and default values can
be shown by running :code:`python trainer.py --help`. Default options can be
changed by providing individual command line arguments. However, it is better
practice to create a configuration file and provide this to the tool. A way
to do this would be:

.. code-block:: bash

    # Dump default configuration to have as reference
    python trainer.py --print_config > default_config.yaml
    # Create config including only options to modify
    nano config.yaml
    # Run training using created configuration
    python trainer.py --config config.yaml

The instantiation of the :class:`~pytorch_lightning.utilities.cli.LightningCLI`
class takes care of parsing command line and config file options, instantiating
the classes, setting up a callback to save the config in the log directory and
finally running :func:`trainer.fit`. The resulting object :code:`cli` can be
used for instance to get the result of fit, i.e., :code:`cli.fit_result`.

After multiple trainings with different configurations, each run will have in
its respective log directory a :code:`config.yaml` file. This file can be used
for reference to know in detail all the settings that were used for each
particular run, and also could be used to trivially reproduce a training, e.g.:

.. code-block:: bash

    python trainer.py --config lightning_logs/version_7/config.yaml

If a separate :class:`~pytorch_lightning.core.datamodule.LightningDataModule`
class is required, the trainer tool just needs a small modification as follows:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(MyModel, MyDataModule)

The start of a possible implementation of :class:`MyModel` including the
recommended argument descriptions in the docstring could be the one below. Note
that by using type hints and docstrings there is no need to duplicate this
information to define its configurable arguments.

.. code-block:: python

    class MyModel(LightningModule):

        def __init__(
            self,
            encoder_layers: int = 12,
            decoder_layers: List[int] = [2, 4]
        ):
            """Example encoder-decoder model

            Args:
                encoder_layers: Number of layers for the encoder
                decoder_layers: Number of layers for each decoder block
            """
            ...

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

The default configuration that option :code:`--print_config` gives is in yaml
format and for the example above would look as follows:

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

Note that there is a section for each class (model and trainer) including all
the init parameters of the class. This grouping is also used in the formatting
of the help shown previously.


Trainer Callbacks and arguments with class type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very important argument of the
:class:`~pytorch_lightning.trainer.trainer.Trainer` class is the
:code:`callbacks`. In contrast to other more simple arguments which just require
numbers or strings, :code:`callbacks` expects a list of instances of subclasses
of :class:`~pytorch_lightning.callbacks.Callback`. To specify this kind of
argument in a config file, each callback must be given as a dictionary including
a :code:`class_path` entry with an import path of the class, and optionally an
:code:`init_args` entry with arguments required to instantiate it. Therefore, a
simple configuration file example that defines a couple of callbacks is the
following:

.. code-block:: yaml

    trainer:
      callbacks:
        - class_path: pytorch_lightning.callbacks.EarlyStopping
          init_args:
            patience: 5
        - class_path: pytorch_lightning.callbacks.LearningRateMonitor
          init_args:
            ...

Similar to the callbacks, any arguments in
:class:`~pytorch_lightning.trainer.trainer.Trainer` and user extended
:class:`~pytorch_lightning.core.lightning.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` classes that
have as type hint a class can be configured the same way using
:code:`class_path` and :code:`init_args`.


Multiple models and/or datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous examples :class:`~pytorch_lightning.utilities.cli.LightningCLI`
works only for a single model and datamodule class. However, there are many
cases in which the objective is to easily be able to run many experiments for
multiple models and datasets. For these cases the tool can be configured such
that a model and/or a datamodule is specified by an import path and init
arguments. For example, with a tool implemented as:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(
        MyModelBaseClass,
        MyDataModuleBaseClass,
        subclass_mode_model=True,
        subclass_mode_data=True
    )

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

Only model classes that are a subclass of :code:`MyModelBaseClass` would be
allowed, and similarly only subclasses of :code:`MyDataModuleBaseClass`.


Customizing LightningCLI
^^^^^^^^^^^^^^^^^^^^^^^^

The init parameters of the
:class:`~pytorch_lightning.utilities.cli.LightningCLI` class can be used to
customize some things, namely: the description of the tool, enabling parsing of
environment variables and additional arguments to instantiate the trainer and
configuration parser.

Nevertheless the init arguments are not enough for many use cases. For this
reason the class is designed so that can be extended to customize different
parts of the command line tool. The argument parser class used by
:class:`~pytorch_lightning.utilities.cli.LightningCLI` is
:class:`~pytorch_lightning.utilities.cli.LightningArgumentParser` which is an
extension of python's argparse, thus adding arguments can be done using the
:func:`add_argument` method. In contrast to argparse it has additional methods
to add arguments, for example :func:`add_class_arguments` adds all arguments
from the init of a class, though requiring parameters to have type hints. For
more details about this please refer to the `respective documentation
<https://omni-us.github.io/jsonargparse/#classes-methods-and-functions>`_.

The :class:`~pytorch_lightning.utilities.cli.LightningCLI` class has the
:meth:`~pytorch_lightning.utilities.cli.LightningCLI.add_arguments_to_parser`
method which can be implemented to include more arguments. After parsing, the
configuration is stored in the :code:`config` attribute of the class instance.
The :class:`~pytorch_lightning.utilities.cli.LightningCLI` class also has two
methods that can be used to run code before and after :code:`trainer.fit` is
executed: :meth:`~pytorch_lightning.utilities.cli.LightningCLI.before_fit` and
:meth:`~pytorch_lightning.utilities.cli.LightningCLI.after_fit`. A realistic
example for these would be to send an email before and after the execution of
fit. The code would be something like:

.. testcode::

    from pytorch_lightning.utilities.cli import LightningCLI

    class MyLightningCLI(LightningCLI):

        def add_arguments_to_parser(self, parser):
            parser.add_argument('--notification_email', default='will@email.com')

        def before_fit(self):
            send_email(
                address=self.config['notification_email'],
                message='trainer.fit starting'
            )

        def after_fit(self):
            send_email(
                address=self.config['notification_email'],
                message='trainer.fit finished'
            )

    cli = MyLightningCLI(MyModel)

Note that the config object :code:`self.config` is a dictionary whose keys are
global options or groups of options. It has the same structure as the yaml
format as described previously. This means for instance that the parameters used
for instantiating the trainer class can be found in
:code:`self.config['trainer']`.

For more advanced use cases, other methods of the
:class:`~pytorch_lightning.utilities.cli.LightningCLI` class could be extended.
For further information have a look at the corresponding API reference.
