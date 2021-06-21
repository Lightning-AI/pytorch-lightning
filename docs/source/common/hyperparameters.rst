.. testsetup:: *

    import torch
    from argparse import ArgumentParser, Namespace
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule
    import sys
    sys.argv = ['foo']

Hyperparameters
---------------
Lightning has utilities to interact seamlessly with the command line ``ArgumentParser``
and plays well with the hyperparameter optimization framework of your choice.

----------

ArgumentParser
^^^^^^^^^^^^^^
Lightning is designed to augment a lot of the functionality of the built-in Python ArgumentParser

.. testcode::

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--layer_1_dim', type=int, default=128)
    args = parser.parse_args()

This allows you to call your program like so:

.. code-block:: bash

    python trainer.py --layer_1_dim 64

----------

Argparser Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^
It is best practice to layer your arguments in three sections.

1.  Trainer args (``gpus``, ``num_nodes``, etc...)
2.  Model specific arguments (``layer_dim``, ``num_layers``, ``learning_rate``, etc...)
3.  Program arguments (``data_path``, ``cluster_email``, etc...)

|

We can do this as follows. First, in your ``LightningModule``, define the arguments
specific to that module. Remember that data splits or data paths may also be specific to
a module (i.e.: if your project has a model that trains on Imagenet and another on CIFAR-10).

.. testcode::

    class LitModel(LightningModule):

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = parent_parser.add_argument_group("LitModel")
            parser.add_argument('--encoder_layers', type=int, default=12)
            parser.add_argument('--data_path', type=str, default='/some/path')
            return parent_parser

Now in your main trainer file, add the ``Trainer`` args, the program args, and add the model args

.. testcode::

    # ----------------
    # trainer_main.py
    # ----------------
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--conda_env', type=str, default='some_name')
    parser.add_argument('--notification_email', type=str, default='will@email.com')

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

Now you can call run your program like so:

.. code-block:: bash

    python trainer_main.py --gpus 2 --num_nodes 2 --conda_env 'my_env' --encoder_layers 12

Finally, make sure to start the training like so:

.. code-block:: python

    # init the trainer like this
    trainer = Trainer.from_argparse_args(args, early_stopping_callback=...)

    # NOT like this
    trainer = Trainer(gpus=hparams.gpus, ...)

    # init the model with Namespace directly
    model = LitModel(args)

    # or init the model with all the key-value pairs
    dict_args = vars(args)
    model = LitModel(**dict_args)

----------

LightningModule hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often times we train many versions of a model. You might share that model or come back to it a few months later
at which point it is very useful to know how that model was trained (i.e.: what learning rate, neural network, etc...).

Lightning has a few ways of saving that information for you in checkpoints and yaml files. The goal here is to
improve readability and reproducibility.

1.  The first way is to ask lightning to save the values of anything in the __init__ for you to the checkpoint. This also
    makes those values available via `self.hparams`.

    .. code-block:: python

        class LitMNIST(LightningModule):

            def __init__(self, layer_1_dim=128, learning_rate=1e-2, **kwargs):
                super().__init__()
                # call this to save (layer_1_dim=128, learning_rate=1e-4) to the checkpoint
                self.save_hyperparameters()

                # equivalent
                self.save_hyperparameters('layer_1_dim', 'learning_rate')

                # Now possible to access layer_1_dim from hparams
                self.hparams.layer_1_dim


2.  Sometimes your init might have objects or other parameters you might not want to save.
    In that case, choose only a few

    .. code-block:: python

        class LitMNIST(LightningModule):

            def __init__(self, loss_fx, generator_network, layer_1_dim=128 **kwargs):
                super().__init__()
                self.layer_1_dim = layer_1_dim
                self.loss_fx = loss_fx

                # call this to save (layer_1_dim=128) to the checkpoint
                self.save_hyperparameters('layer_1_dim')

        # to load specify the other args
        model = LitMNIST.load_from_checkpoint(PATH, loss_fx=torch.nn.SomeOtherLoss, generator_network=MyGenerator())


3.  You can also save full objects such as `dict` or `Namespace` to the checkpoint.

    .. code-block:: python

        # using a argparse.Namespace
        class LitMNIST(LightningModule):

            def __init__(self, conf, *args, **kwargs):
                super().__init__()
                self.save_hyperparameters(conf)

                self.layer_1 = nn.Linear(28 * 28, self.hparams.layer_1_dim)
                self.layer_2 = nn.Linear(self.hparams.layer_1_dim, self.hparams.layer_2_dim)
                self.layer_3 = nn.Linear(self.hparams.layer_2_dim, 10)

        conf = OmegaConf.create(...)
        model = LitMNIST(conf)

        # Now possible to access any stored variables from hparams
        model.hparams.anything



----------

Trainer args
^^^^^^^^^^^^
To recap, add ALL possible trainer flags to the argparser and init the ``Trainer`` this way

.. code-block:: python

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    trainer = Trainer.from_argparse_args(hparams)

    # or if you need to pass in callbacks
    trainer = Trainer.from_argparse_args(hparams, checkpoint_callback=..., callbacks=[...])

----------

Multiple Lightning Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

We often have multiple Lightning Modules where each one has different arguments. Instead of
polluting the ``main.py`` file, the ``LightningModule`` lets you define arguments for each one.

.. testcode::

    class LitMNIST(LightningModule):

        def __init__(self, layer_1_dim, **kwargs):
            super().__init__()
            self.layer_1 = nn.Linear(28 * 28, layer_1_dim)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = parent_parser.add_argument_group("LitMNIST")
            parser.add_argument('--layer_1_dim', type=int, default=128)
            return parent_parser

.. testcode::

    class GoodGAN(LightningModule):

        def __init__(self, encoder_layers, **kwargs):
            super().__init__()
            self.encoder = Encoder(layers=encoder_layers)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = parent_parser.add_argument_group("GoodGAN")
            parser.add_argument('--encoder_layers', type=int, default=12)
            return parent_parser


Now we can allow each model to inject the arguments it needs in the ``main.py``

.. code-block:: python

    def main(args):
        dict_args = vars(args)

        # pick model
        if args.model_name == 'gan':
            model = GoodGAN(**dict_args)
        elif args.model_name == 'mnist':
            model = LitMNIST(**dict_args)

        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model)

    if __name__ == '__main__':
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)

        # figure out which model to use
        parser.add_argument('--model_name', type=str, default='gan', help='gan or mnist')

        # THIS LINE IS KEY TO PULL THE MODEL NAME
        temp_args, _ = parser.parse_known_args()

        # let the model add what it wants
        if temp_args.model_name == 'gan':
            parser = GoodGAN.add_model_specific_args(parser)
        elif temp_args.model_name == 'mnist':
            parser = LitMNIST.add_model_specific_args(parser)

        args = parser.parse_args()

        # train
        main(args)

and now we can train MNIST or the GAN using the command line interface!

.. code-block:: bash

    $ python main.py --model_name gan --encoder_layers 24
    $ python main.py --model_name mnist --layer_1_dim 128
