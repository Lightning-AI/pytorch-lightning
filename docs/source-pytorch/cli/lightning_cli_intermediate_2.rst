:orphan:

###########################################
Eliminate config boilerplate (intermediate)
###########################################
**Audience:** Users who have multiple models and datasets per project.

**Pre-reqs:** You must have read :doc:`(Control it all from the CLI) <lightning_cli_intermediate>`.

----

****************************************
Why do I want to mix models and datasets
****************************************
Lightning projects usually begin with one model and one dataset. As the project grows in complexity and you introduce more models and more datasets, it becomes desirable
to mix any model with any dataset directly from the commandline without changing your code.


.. code:: bash

    # Mix and match anything
    $ python main.py fit --model=GAN --data=MNIST
    $ python main.py fit --model=Transformer --data=MNIST

This is what the Lightning CLI enables. Otherwise, this kind of configuration requires a significant amount of boilerplate that often looks like this:

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

----

*************************
Register LightningModules
*************************
Connect models across different files with the ``MODEL_REGISTRY`` to make them available from the CLI:

.. code:: python

    # main.py

    from pytorch_lightning import demos
    from pytorch_lightning.utilities import cli as pl_cli


    @pl_cli.MODEL_REGISTRY
    class Model1(DemoModel):
        def configure_optimizers(self):
            print("⚡", "using Model1", "⚡")
            return super().configure_optimizers()


    @pl_cli.MODEL_REGISTRY
    class Model2(DemoModel):
        def configure_optimizers(self):
            print("⚡", "using Model2", "⚡")
            return super().configure_optimizers()


    cli = pl_cli.LightningCLI(datamodule_class=BoringDataModule)

Now you can choose between any model from the CLI:

.. code:: bash

    # use Model1
    python main.py fit --model Model1

    # use Model2
    python main.py fit --model Model2

----

********************
Register DataModules
********************
Connect DataModules across different files with the ``DATAMODULE_REGISTRY`` to make them available from the CLI:

.. code:: python

    # main.py
    import torch
    from pytorch_lightning.utilities import cli as pl_cli
    from pytorch_lightning import demos


    @pl_cli.DATAMODULE_REGISTRY
    class FakeDataset1(BoringDataModule):
        def train_dataloader(self):
            print("⚡", "using FakeDataset1", "⚡")
            return torch.utils.data.DataLoader(self.random_train)


    @pl_cli.DATAMODULE_REGISTRY
    class FakeDataset2(BoringDataModule):
        def train_dataloader(self):
            print("⚡", "using FakeDataset2", "⚡")
            return torch.utils.data.DataLoader(self.random_train)


    cli = pl_cli.LightningCLI(DemoModel)

Now you can choose between any dataset at runtime:

.. code:: bash

    # use Model1
    python main.py fit --data FakeDataset1

    # use Model2
    python main.py fit --data FakeDataset2

----

*******************
Register optimizers
*******************
Connect optimizers with the ``OPTIMIZER_REGISTRY`` to make them available from the CLI:

.. code:: python

    # main.py
    import torch
    from pytorch_lightning.utilities import cli as pl_cli
    from pytorch_lightning import demos


    @pl_cli.OPTIMIZER_REGISTRY
    class LitAdam(torch.optim.Adam):
        def step(self, closure):
            print("⚡", "using LitAdam", "⚡")
            super().step(closure)


    @pl_cli.OPTIMIZER_REGISTRY
    class FancyAdam(torch.optim.Adam):
        def step(self, closure):
            print("⚡", "using FancyAdam", "⚡")
            super().step(closure)


    cli = pl_cli.LightningCLI(DemoModel, BoringDataModule)

Now you can choose between any optimizer at runtime:

.. code:: bash

    # use LitAdam
    python main.py fit --optimizer LitAdam

    # use FancyAdam
    python main.py fit --optimizer FancyAdam

Bonus: If you need only 1 optimizer, the Lightning CLI already works out of the box with any Optimizer from ``torch.optim.optim``:

.. code:: bash

    python main.py fit --optimizer AdamW

If the optimizer you want needs other arguments, add them via the CLI (no need to change your code)!

.. code:: bash

    python main.py fit --optimizer SGD --optimizer.lr=0.01

----

**********************
Register LR schedulers
**********************
Connect learning rate schedulers with the ``LR_SCHEDULER_REGISTRY`` to make them available from the CLI:

.. code:: python

    # main.py
    import torch
    from pytorch_lightning.utilities import cli as pl_cli
    from pytorch_lightning import demos


    @pl_cli.LR_SCHEDULER_REGISTRY
    class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
        def step(self):
            print("⚡", "using LitLRScheduler", "⚡")
            super().step()


    cli = pl_cli.LightningCLI(DemoModel, BoringDataModule)

Now you can choose between any learning rate scheduler at runtime:

.. code:: bash

    # LitLRScheduler
    python main.py fit --lr_scheduler LitLRScheduler


Bonus: If you need only 1 LRScheduler, the Lightning CLI already works out of the box with any LRScheduler from ``torch.optim``:

.. code:: bash

    python main.py fit --lr_scheduler CosineAnnealingLR
    python main.py fit --lr_scheduler LinearLR
    ...

If the scheduler you want needs other arguments, add them via the CLI (no need to change your code)!

.. code:: bash

    python main.py fit --lr_scheduler=ReduceLROnPlateau --lr_scheduler.monitor=epoch

----

*************************
Register from any package
*************************
A shortcut to register many classes from a package is to use the ``register_classes`` method. Here we register all optimizers from the ``torch.optim`` library:

.. code:: python

    import torch
    from pytorch_lightning.utilities import cli as pl_cli
    from pytorch_lightning import demos

    # add all PyTorch optimizers!
    pl_cli.OPTIMIZER_REGISTRY.register_classes(module=torch.optim, base_cls=torch.optim.Optimizer)

    cli = pl_cli.LightningCLI(DemoModel, BoringDataModule)

Now use any of the optimizers in the ``torch.optim`` library:

.. code:: bash

    python main.py fit --optimizer AdamW

This method is supported by all the registry classes.
