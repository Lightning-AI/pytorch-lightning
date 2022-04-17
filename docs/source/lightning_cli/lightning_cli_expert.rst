#######################################
Eliminate config boilerplate (Advanced)
#######################################



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
and after the trainer runs: :code:`before_<subcommand>` and :code:`after_<subcommand>`.
A realistic example for these would be to send an email before and after the execution.
The code for the :code:`fit` subcommand would be something like:

.. testcode::

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
instantiating the trainer class can be found in :code:`self.config['fit']['trainer']`.

.. tip::

    Have a look at the :class:`~pytorch_lightning.utilities.cli.LightningCLI` class API reference to learn about other
    methods that can be extended to customize a CLI.


Configurable callbacks
^^^^^^^^^^^^^^^^^^^^^^

As explained previously, any Lightning callback can be added by passing it through command line or
including it in the config via :code:`class_path` and :code:`init_args` entries.
However, there are other cases in which a callback should always be present and be configurable.
This can be implemented as follows:

.. testcode::

    from pytorch_lightning.callbacks import EarlyStopping


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_lightning_class_args(EarlyStopping, "my_early_stopping")
            parser.set_defaults({"my_early_stopping.monitor": "val_loss", "my_early_stopping.patience": 5})


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

***********************
Create your own command
***********************
    carlos:
        trainer:
            limit_train_batches: 100
            max_epochs: 10
    test:
        trainer:
            limit_test_batches: 10

python main.py --config a.yaml carlos