:orphan:

.. testsetup:: *
    :skipif: not _JSONARGPARSE_AVAILABLE

    import torch
    from unittest import mock
    from typing import List
    import lightning.pytorch.cli as pl_cli
    from lightning.pytorch import LightningModule, LightningDataModule, Trainer, Callback


    class NoFitTrainer(Trainer):
        def fit(self, *_, **__):
            pass


    class LightningCLI(pl_cli.LightningCLI):
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

#################################################
Configure hyperparameters from the CLI (Advanced)
#################################################

Instantiation only mode
^^^^^^^^^^^^^^^^^^^^^^^

The CLI is designed to start fitting with minimal code changes. On class instantiation, the CLI will automatically call
the trainer function associated with the subcommand provided, so you don't have to do it. To avoid this, you can set the
following argument:

.. testcode::

    cli = LightningCLI(MyModel, run=False)  # True by default
    # you'll have to call fit yourself:
    cli.trainer.fit(cli.model)

In this mode, subcommands are **not** added to the parser. This can be useful to implement custom logic without having
to subclass the CLI, but still, use the CLI's instantiation and argument parsing capabilities.


Trainer Callbacks and arguments with class type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very important argument of the :class:`~lightning.pytorch.trainer.trainer.Trainer` class is the ``callbacks``. In
contrast to simpler arguments that take numbers or strings, ``callbacks`` expects a list of instances of subclasses of
:class:`~lightning.pytorch.callbacks.Callback`. To specify this kind of argument in a config file, each callback must be
given as a dictionary, including a ``class_path`` entry with an import path of the class and optionally an ``init_args``
entry with arguments to use to instantiate. Therefore, a simple configuration file that defines two callbacks is the
following:

.. code-block:: yaml

    trainer:
      callbacks:
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            save_weights_only: true
        - class_path: lightning.pytorch.callbacks.LearningRateMonitor
          init_args:
            logging_interval: 'epoch'

Similar to the callbacks, any parameter in :class:`~lightning.pytorch.trainer.trainer.Trainer` and user extended
:class:`~lightning.pytorch.core.LightningModule` and
:class:`~lightning.pytorch.core.datamodule.LightningDataModule` classes that have as type hint a class, can be
configured the same way using ``class_path`` and ``init_args``. If the package that defines a subclass is imported
before the :class:`~lightning.pytorch.cli.LightningCLI` class is run, the name can be used instead of the full import
path.

From command line the syntax is the following:

.. code-block:: bash

    $ python ... \
        --trainer.callbacks+={CALLBACK_1_NAME} \
        --trainer.callbacks.{CALLBACK_1_ARGS_1}=... \
        --trainer.callbacks.{CALLBACK_1_ARGS_2}=... \
        ...
        --trainer.callbacks+={CALLBACK_N_NAME} \
        --trainer.callbacks.{CALLBACK_N_ARGS_1}=... \
        ...

Note the use of ``+`` to append a new callback to the list and that the ``init_args`` are applied to the previous
callback appended. Here is an example:

.. code-block:: bash

    $ python ... \
        --trainer.callbacks+=EarlyStopping \
        --trainer.callbacks.patience=5 \
        --trainer.callbacks+=LearningRateMonitor \
        --trainer.callbacks.logging_interval=epoch

.. note::

    Serialized config files (e.g. ``--print_config`` or :class:`~lightning.pytorch.cli.SaveConfigCallback`) always have
    the full ``class_path``, even when class name shorthand notation is used in the command line or in input config
    files.


Multiple models and/or datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A CLI can be written such that a model and/or a datamodule is specified by an import path and init arguments. For
example, with a tool implemented as:

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
        - class_path: lightning.pytorch.callbacks.EarlyStopping
          init_args:
            patience: 5
        ...

Only model classes that are a subclass of ``MyModelBaseClass`` would be allowed, and similarly, only subclasses of
``MyDataModuleBaseClass``. If as base classes :class:`~lightning.pytorch.core.LightningModule` and
:class:`~lightning.pytorch.core.datamodule.LightningDataModule` is given, then the CLI would allow any lightning module
and data module.

.. tip::

    Note that with the subclass modes, the ``--help`` option does not show information for a specific subclass. To get
    help for a subclass, the options ``--model.help`` and ``--data.help`` can be used, followed by the desired class
    path. Similarly, ``--print_config`` does not include the settings for a particular subclass. To include them, the
    class path should be given before the ``--print_config`` option. Examples for both help and print config are:

    .. code-block:: bash

        $ python trainer.py fit --model.help mycode.mymodels.MyModel
        $ python trainer.py fit --model mycode.mymodels.MyModel --print_config


Models with multiple submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many use cases require to have several modules, each with its own configurable options. One possible way to handle this
with ``LightningCLI`` is to implement a single module having as init parameters each of the submodules. This is known as
`dependency injection <https://en.wikipedia.org/wiki/Dependency_injection>`__ which is a good approach to improve
decoupling in your code base.

Since the init parameters of the model have as a type hint a class, in the configuration, these would be specified with
``class_path`` and ``init_args`` entries. For instance, a model could be implemented as:

.. testcode::

    class MyMainModel(LightningModule):
        def __init__(self, encoder: nn.Module, decoder: nn.Module):
            """Example encoder-decoder submodules model

            Args:
                encoder: Instance of a module for encoding
                decoder: Instance of a module for decoding
            """
            super().__init__()
            self.save_hyperparameters()
            self.encoder = encoder
            self.decoder = decoder

If the CLI is implemented as ``LightningCLI(MyMainModel)`` the configuration would be as follows:

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

It is also possible to combine ``subclass_mode_model=True`` and submodules, thereby having two levels of ``class_path``.

.. tip::

    By having ``self.save_hyperparameters()`` it becomes possible to load the model from a checkpoint. Simply do
    ``ModelClass.load_from_checkpoint("path/to/checkpoint.ckpt")``. In the case of using ``subclass_mode_model=True``,
    then load it like ``LightningModule.load_from_checkpoint("path/to/checkpoint.ckpt")``. ``save_hyperparameters`` is
    optional and can be safely removed if there is no need to load from a checkpoint.


Fixed optimizer and scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, fixing the optimizer and/or learning scheduler might be desired instead of allowing multiple. For this,
you can manually add the arguments for specific classes by subclassing the CLI. The following code snippet shows how to
implement it:

.. testcode::

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

With this, in the config, the ``optimizer`` and ``lr_scheduler`` groups would accept all of the options for the given
classes, in this example, ``Adam`` and ``ExponentialLR``. Therefore, the config file would be structured like:

.. code-block:: yaml

    optimizer:
      lr: 0.01
    lr_scheduler:
      gamma: 0.2
    model:
      ...
    trainer:
      ...

where the arguments can be passed directly through the command line without specifying the class. For example:

.. code-block:: bash

    $ python trainer.py fit --optimizer.lr=0.01 --lr_scheduler.gamma=0.2


Multiple optimizers and schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the CLIs support multiple optimizers and/or learning schedulers, automatically implementing
``configure_optimizers``. This behavior can be disabled by providing ``auto_configure_optimizers=False`` on
instantiation of :class:`~lightning.pytorch.cli.LightningCLI`. This would be required for example to support multiple
optimizers, for each selecting a particular optimizer class. Similar to multiple submodules, this can be done via
`dependency injection <https://en.wikipedia.org/wiki/Dependency_injection>`__. Unlike the submodules, it is not possible
to expect an instance of a class, because optimizers require the module's parameters to optimize, which are only
available after instantiation of the module. Learning schedulers are a similar situation, requiring an optimizer
instance. For these cases, dependency injection involves providing a function that instantiates the respective class
when called.

An example of a model that uses two optimizers is the following:

.. code-block:: python

    from typing import Iterable
    from torch.optim import Optimizer


    OptimizerCallable = Callable[[Iterable], Optimizer]


    class MyModel(LightningModule):
        def __init__(self, optimizer1: OptimizerCallable, optimizer2: OptimizerCallable):
            super().__init__()
            self.save_hyperparameters()
            self.optimizer1 = optimizer1
            self.optimizer2 = optimizer2

        def configure_optimizers(self):
            optimizer1 = self.optimizer1(self.parameters())
            optimizer2 = self.optimizer2(self.parameters())
            return [optimizer1, optimizer2]


    cli = MyLightningCLI(MyModel, auto_configure_optimizers=False)

Note the type ``Callable[[Iterable], Optimizer]``, which denotes a function that receives a single argument, some
learnable parameters, and returns an optimizer instance. With this, from the command line it is possible to select the
class and init arguments for each of the optimizers, as follows:

.. code-block:: bash

    $ python trainer.py fit \
        --model.optimizer1=Adam \
        --model.optimizer1.lr=0.01 \
        --model.optimizer2=AdamW \
        --model.optimizer2.lr=0.0001

In the example above, the ``OptimizerCallable`` type alias was created to illustrate what the type hint means. For
convenience, this type alias and one for learning schedulers is available in the ``cli`` module. An example of a model
that uses dependency injection for an optimizer and a learning scheduler is:

.. code-block:: python

    from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, LightningCLI


    class MyModel(LightningModule):
        def __init__(
            self,
            optimizer: OptimizerCallable = torch.optim.Adam,
            scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        ):
            super().__init__()
            self.save_hyperparameters()
            self.optimizer = optimizer
            self.scheduler = scheduler

        def configure_optimizers(self):
            optimizer = self.optimizer(self.parameters())
            scheduler = self.scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}


    cli = MyLightningCLI(MyModel, auto_configure_optimizers=False)

Note that for this example, classes are used as defaults. This is compatible with the type hints, since they are also
callables that receive the same first argument and return an instance of the class. Classes that have more than one
required argument will not work as default. For these cases a lambda function can be used, e.g. ``optimizer:
OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01)``.


Run from Python
^^^^^^^^^^^^^^^

Even though the :class:`~lightning.pytorch.cli.LightningCLI` class is designed to help in the implementation of command
line tools, for some use cases it is desired to run directly from Python. To allow this there is the ``args`` parameter.
An example could be to first implement a normal CLI script, but adding an ``args`` parameter with default ``None`` to
the main function as follows:

.. code:: python

    from lightning.pytorch.cli import ArgsType, LightningCLI


    def cli_main(args: ArgsType = None):
        cli = LightningCLI(MyModel, ..., args=args)
        ...


    if __name__ == "__main__":
        cli_main()

Then it is possible to import the ``cli_main`` function to run it. Executing in a shell ``my_cli.py
--trainer.max_epochs=100 --model.encoder_layers=24`` would be equivalent to:

.. code:: python

    from my_module.my_cli import cli_main

    cli_main(["--trainer.max_epochs=100", "--model.encoder_layers=24"])

All the features that are supported from the command line can be used when giving ``args`` as a list of strings. It is
also possible to provide a ``dict`` or `jsonargparse.Namespace
<https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.Namespace>`__. For example in a jupyter notebook someone
might do:

.. code:: python

    args = {
        "trainer": {
            "max_epochs": 100,
        },
        "model": {},
    }

    args["model"]["encoder_layers"] = 8
    cli_main(args)
    args["model"]["encoder_layers"] = 12
    cli_main(args)
    args["trainer"]["max_epochs"] = 200
    cli_main(args)

.. note::

    The ``args`` parameter must be ``None`` when running from command line so that ``sys.argv`` is used as arguments.
    Also, note that the purpose of ``trainer_defaults`` is different to ``args``. It is okay to use ``trainer_defaults``
    in the ``cli_main`` function to modify the defaults of some trainer parameters.
