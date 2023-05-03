:orphan:

#####################################################
Configure hyperparameters from the CLI (Intermediate)
#####################################################
**Audience:** Users who have multiple models and datasets per project.

**Pre-reqs:** You must have read :doc:`(Control it all from the CLI) <lightning_cli_intermediate>`.

----

***************************
Why mix models and datasets
***************************
Lightning projects usually begin with one model and one dataset. As the project grows in complexity and you introduce
more models and more datasets, it becomes desirable to mix any model with any dataset directly from the command line
without changing your code.

.. code:: bash

    # Mix and match anything
    $ python main.py fit --model=GAN --data=MNIST
    $ python main.py fit --model=Transformer --data=MNIST

``LightningCLI`` makes this very simple. Otherwise, this kind of configuration requires a significant amount of
boilerplate that often looks like this:

.. code:: python

    # choose model
    if args.model == "gan":
        model = GAN(args.feat_dim)
    elif args.model == "transformer":
        model = Transformer(args.feat_dim)
    ...

    # choose datamodule
    if args.data == "MNIST":
        datamodule = MNIST()
    elif args.data == "imagenet":
        datamodule = Imagenet()
    ...

    # mix them!
    trainer.fit(model, datamodule)

It is highly recommended that you avoid writing this kind of boilerplate and use ``LightningCLI`` instead.

----

*************************
Multiple LightningModules
*************************
To support multiple models, when instantiating ``LightningCLI`` omit the ``model_class`` parameter:

.. code:: python

    # main.py
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


    class Model1(DemoModel):
        def configure_optimizers(self):
            print("⚡", "using Model1", "⚡")
            return super().configure_optimizers()


    class Model2(DemoModel):
        def configure_optimizers(self):
            print("⚡", "using Model2", "⚡")
            return super().configure_optimizers()


    cli = LightningCLI(datamodule_class=BoringDataModule)

Now you can choose between any model from the CLI:

.. code:: bash

    # use Model1
    python main.py fit --model Model1

    # use Model2
    python main.py fit --model Model2

.. tip::

    Instead of omitting the ``model_class`` parameter, you can give a base class and ``subclass_mode_model=True``. This
    will make the CLI only accept models which are a subclass of the given base class.

----

*****************************
Multiple LightningDataModules
*****************************
To support multiple data modules, when instantiating ``LightningCLI`` omit the ``datamodule_class`` parameter:

.. code:: python

    # main.py
    import torch
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


    class FakeDataset1(BoringDataModule):
        def train_dataloader(self):
            print("⚡", "using FakeDataset1", "⚡")
            return torch.utils.data.DataLoader(self.random_train)


    class FakeDataset2(BoringDataModule):
        def train_dataloader(self):
            print("⚡", "using FakeDataset2", "⚡")
            return torch.utils.data.DataLoader(self.random_train)


    cli = LightningCLI(DemoModel)

Now you can choose between any dataset at runtime:

.. code:: bash

    # use Model1
    python main.py fit --data FakeDataset1

    # use Model2
    python main.py fit --data FakeDataset2

.. tip::

    Instead of omitting the ``datamodule_class`` parameter, you can give a base class and ``subclass_mode_data=True``.
    This will make the CLI only accept data modules that are a subclass of the given base class.

----

*******************
Multiple optimizers
*******************
Standard optimizers from ``torch.optim`` work out of the box:

.. code:: bash

    python main.py fit --optimizer AdamW

If the optimizer you want needs other arguments, add them via the CLI (no need to change your code)!

.. code:: bash

    python main.py fit --optimizer SGD --optimizer.lr=0.01

Furthermore, any custom subclass of :class:`torch.optim.Optimizer` can be used as an optimizer:

.. code:: python

    # main.py
    import torch
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


    class LitAdam(torch.optim.Adam):
        def step(self, closure):
            print("⚡", "using LitAdam", "⚡")
            super().step(closure)


    class FancyAdam(torch.optim.Adam):
        def step(self, closure):
            print("⚡", "using FancyAdam", "⚡")
            super().step(closure)


    cli = LightningCLI(DemoModel, BoringDataModule)

Now you can choose between any optimizer at runtime:

.. code:: bash

    # use LitAdam
    python main.py fit --optimizer LitAdam

    # use FancyAdam
    python main.py fit --optimizer FancyAdam

----

*******************
Multiple schedulers
*******************
Standard learning rate schedulers from ``torch.optim.lr_scheduler``  work out of the box:

.. code:: bash

    python main.py fit --optimizer=Adam --lr_scheduler CosineAnnealingLR

Please note that ``--optimizer`` must be added for ``--lr_scheduler`` to have an effect.

If the scheduler you want needs other arguments, add them via the CLI (no need to change your code)!

.. code:: bash

    python main.py fit --optimizer=Adam --lr_scheduler=ReduceLROnPlateau --lr_scheduler.monitor=epoch

Furthermore, any custom subclass of ``torch.optim.lr_scheduler.LRScheduler`` can be used as learning rate scheduler:

.. code:: python

    # main.py
    import torch
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


    class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
        def step(self):
            print("⚡", "using LitLRScheduler", "⚡")
            super().step()


    cli = LightningCLI(DemoModel, BoringDataModule)

Now you can choose between any learning rate scheduler at runtime:

.. code:: bash

    # LitLRScheduler
    python main.py fit --optimizer=Adam --lr_scheduler LitLRScheduler


----

************************
Classes from any package
************************
In the previous sections, custom classes to select were defined in the same python file where the ``LightningCLI`` class
is run. To select classes from any package by using only the class name, import the respective package:

.. code:: python

    from lightning.pytorch.cli import LightningCLI
    import my_code.models  # noqa: F401
    import my_code.data_modules  # noqa: F401
    import my_code.optimizers  # noqa: F401

    cli = LightningCLI()

Now use any of the classes:

.. code:: bash

    python main.py fit --model Model1 --data FakeDataset1 --optimizer LitAdam --lr_scheduler LitLRScheduler

The ``# noqa: F401`` comment avoids a linter warning that the import is unused.

It is also possible to select subclasses that have not been imported by giving the full import path:

.. code:: bash

    python main.py fit --model my_code.models.Model1

----

*************************
Help for specific classes
*************************
When multiple models or datasets are accepted, the main help of the CLI does not include their specific parameters. To
show this specific help, additional help arguments expect the class name or its import path. For example:

.. code:: bash

    python main.py fit --model.help Model1
    python main.py fit --data.help FakeDataset2
    python main.py fit --optimizer.help Adagrad
    python main.py fit --lr_scheduler.help StepLR
