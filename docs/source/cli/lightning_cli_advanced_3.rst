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


    MyModelBaseClass = MyModel
    MyDataModuleBaseClass = MyDataModule

    mock_argv = mock.patch("sys.argv", ["any.py"])
    mock_argv.start()

.. testcleanup:: *

    mock_argv.stop()

Instantiation only mode
^^^^^^^^^^^^^^^^^^^^^^^

The CLI is designed to start fitting with minimal code changes. On class instantiation, the CLI will automatically
call the trainer function associated to the subcommand provided so you don't have to do it.
To avoid this, you can set the following argument:

.. testcode::

    cli = LightningCLI(MyModel, run=False)  # True by default
    # you'll have to call fit yourself:
    cli.trainer.fit(cli.model)

In this mode, there are subcommands added to the parser.
This can be useful to implement custom logic without having to subclass the CLI, but still using the CLI's instantiation
and argument parsing capabilities.


Subclass registration
^^^^^^^^^^^^^^^^^^^^^

To use shorthand notation, the options need to be registered beforehand. This can be easily done with:

.. code-block::

    LightningCLI(auto_registry=True)  # False by default

which will register all subclasses of :class:`torch.optim.Optimizer`, :class:`torch.optim.lr_scheduler._LRScheduler`,
:class:`~pytorch_lightning.core.module.LightningModule`,
:class:`~pytorch_lightning.core.datamodule.LightningDataModule`, :class:`~pytorch_lightning.callbacks.Callback`, and
:class:`~pytorch_lightning.loggers.LightningLoggerBase` across all imported modules. This includes those in your own
code.

Alternatively, if this is left unset, only the subclasses defined in PyTorch's :class:`torch.optim.Optimizer`,
:class:`torch.optim.lr_scheduler._LRScheduler` and Lightning's :class:`~pytorch_lightning.callbacks.Callback` and
:class:`~pytorch_lightning.loggers.LightningLoggerBase` subclassess will be registered.

In subsequent sections, we will go over adding specific classes to specific registries as well as how to use
shorthand notation.


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
:class:`~pytorch_lightning.core.module.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` classes that have as type hint a class can be configured
the same way using :code:`class_path` and :code:`init_args`.

For callbacks in particular, Lightning simplifies the command line so that only
the :class:`~pytorch_lightning.callbacks.Callback` name is required.
The argument's order matters and the user needs to pass the arguments in the following way.

.. code-block:: bash

    $ python ... \
        --trainer.callbacks+={CALLBACK_1_NAME} \
        --trainer.callbacks.{CALLBACK_1_ARGS_1}=... \
        --trainer.callbacks.{CALLBACK_1_ARGS_2}=... \
        ...
        --trainer.callbacks+={CALLBACK_N_NAME} \
        --trainer.callbacks.{CALLBACK_N_ARGS_1}=... \
        ...

Here is an example:

.. code-block:: bash

    $ python ... \
        --trainer.callbacks+=EarlyStopping \
        --trainer.callbacks.patience=5 \
        --trainer.callbacks+=LearningRateMonitor \
        --trainer.callbacks.logging_interval=epoch

Lightning provides a mechanism for you to add your own callbacks and benefit from the command line simplification
as described above:

.. code-block:: python

    from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


    @CALLBACK_REGISTRY
    class CustomCallback(Callback):
        ...


    cli = LightningCLI(...)

.. code-block:: bash

    $  python ... --trainer.callbacks+=CustomCallback ...

.. note::

    This shorthand notation is also supported inside a configuration file. The configuration file
    generated by calling the previous command with ``--print_config`` will have the full ``class_path`` notation.

    .. code-block:: yaml

        trainer:
          callbacks:
            - class_path: your_class_path.CustomCallback
              init_args:
                ...


.. tip::

    ``--trainer.logger`` also supports shorthand notation and a ``LOGGER_REGISTRY`` is available to register custom
    Loggers.


Multiple models and/or datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, the tool can be configured such that a model and/or a datamodule is
specified by an import path and init arguments. For example, with a tool implemented as:

.. code-block:: python

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
:code:`MyDataModuleBaseClass`. If as base classes :class:`~pytorch_lightning.core.module.LightningModule` and
:class:`~pytorch_lightning.core.datamodule.LightningDataModule` are given, then the tool would allow any lightning
module and data module.

.. tip::

    Note that with the subclass modes the :code:`--help` option does not show information for a specific subclass. To
    get help for a subclass the options :code:`--model.help` and :code:`--data.help` can be used, followed by the
    desired class path. Similarly :code:`--print_config` does not include the settings for a particular subclass. To
    include them the class path should be given before the :code:`--print_config` option. Examples for both help and
    print config are:

    .. code-block:: bash

        $ python trainer.py fit --model.help mycode.mymodels.MyModel
        $ python trainer.py fit --model mycode.mymodels.MyModel --print_config


Models with multiple submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many use cases require to have several modules each with its own configurable options. One possible way to handle this
with LightningCLI is to implement a single module having as init parameters each of the submodules. Since the init
parameters have as type a class, then in the configuration these would be specified with :code:`class_path` and
:code:`init_args` entries. For instance a model could be implemented as:

.. testcode::

    class MyMainModel(LightningModule):
        def __init__(self, encoder: nn.Module, decoder: nn.Module):
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


Class type defaults
^^^^^^^^^^^^^^^^^^^

The support for classes as type hints allows to try many possibilities with the same CLI. This is a useful feature, but
it can make it tempting to use an instance of a class as a default. For example:

.. code-block::

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

Optimizers
^^^^^^^^^^

If you will not be changing the class, you can manually add the arguments for specific optimizers and/or
learning rate schedulers by subclassing the CLI. This has the advantage of providing the proper help message for those
classes. The following code snippet shows how to implement it:

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

With this, in the config the :code:`optimizer` and :code:`lr_scheduler` groups would accept all of the options for the
given classes, in this example :code:`Adam` and :code:`ExponentialLR`.
Therefore, the config file would be structured like:

.. code-block:: yaml

    optimizer:
      lr: 0.01
    lr_scheduler:
      gamma: 0.2
    model:
      ...
    trainer:
      ...

Where the arguments can be passed directly through command line without specifying the class. For example:

.. code-block:: bash

    $ python trainer.py fit --optimizer.lr=0.01 --lr_scheduler.gamma=0.2

The automatic implementation of :code:`configure_optimizers` can be disabled by linking the configuration group. An
example can be when one wants to add support for multiple optimizers:

.. code-block:: python

    from pytorch_lightning.utilities.cli import instantiate_class


    class MyModel(LightningModule):
        def __init__(self, optimizer1_init: dict, optimizer2_init: dict):
            super().__init__()
            self.optimizer1_init = optimizer1_init
            self.optimizer2_init = optimizer2_init

        def configure_optimizers(self):
            optimizer1 = instantiate_class(self.parameters(), self.optimizer1_init)
            optimizer2 = instantiate_class(self.parameters(), self.optimizer2_init)
            return [optimizer1, optimizer2]


    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(
                OPTIMIZER_REGISTRY.classes, nested_key="gen_optimizer", link_to="model.optimizer1_init"
            )
            parser.add_optimizer_args(
                OPTIMIZER_REGISTRY.classes, nested_key="gen_discriminator", link_to="model.optimizer2_init"
            )


    cli = MyLightningCLI(MyModel)

The value given to :code:`optimizer*_init` will always be a dictionary including :code:`class_path` and
:code:`init_args` entries. The function :func:`~pytorch_lightning.utilities.cli.instantiate_class`
takes care of importing the class defined in :code:`class_path` and instantiating it using some positional arguments,
in this case :code:`self.parameters()`, and the :code:`init_args`.
Any number of optimizers and learning rate schedulers can be added when using :code:`link_to`.

With shorthand notation:

.. code-block:: bash

    $ python trainer.py fit \
        --gen_optimizer=Adam \
        --gen_optimizer.lr=0.01 \
        --gen_discriminator=AdamW \
        --gen_discriminator.lr=0.0001

You can also pass the class path directly, for example, if the optimizer hasn't been registered to the
``OPTIMIZER_REGISTRY``:

.. code-block:: bash

    $ python trainer.py fit \
        --gen_optimizer.class_path=torch.optim.Adam \
        --gen_optimizer.init_args.lr=0.01 \
        --gen_discriminator.class_path=torch.optim.AdamW \
        --gen_discriminator.init_args.lr=0.0001
