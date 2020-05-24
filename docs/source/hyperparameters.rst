.. testsetup:: *

    import torch
    from argparse import ArgumentParser, Namespace
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule
    import sys
    sys.argv = ['foo']


Hyperparameters
---------------
Lightning has utilities to interact seamlessly with the command line ArgumentParser
and plays well with the hyperparameter optimization framework of your choice.

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


Argparser Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^
It is best practice to layer your arguments in three sections.

1.  Trainer args (gpus, num_nodes, etc...)
2.  Model specific arguments (layer_dim, num_layers, learning_rate, etc...)
3.  Program arguments (data_path, cluster_email, etc...)

We can do this as follows. First, in your LightningModule, define the arguments
specific to that module. Remember that data splits or data paths may also be specific to
a module (ie: if your project has a model that trains on Imagenet and another on CIFAR-10).

.. testcode::

    class LitModel(LightningModule):

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--encoder_layers', type=int, default=12)
            parser.add_argument('--data_path', type=str, default='/some/path')
            return parser

Now in your main trainer file, add the Trainer args, the program args, and add the model args

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

Now you can call run your program like so

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

LightningModule hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: The use of `hparams` is no longer recommended (but still supported)

LightningModule is just an nn.Module, you can use it as you normally would. However, there are
some best practices to improve readability and reproducibility.

1. It's more readable to specify all the arguments that go into a module (with default values).
This helps users of your module know everything that is required to run this.

.. testcode::

    class LitMNIST(LightningModule):

        def __init__(self, layer_1_dim=128, layer_2_dim=256, learning_rate=1e-4, batch_size=32, **kwargs):
            super().__init__()
            self.layer_1_dim = layer_1_dim
            self.layer_2_dim = layer_2_dim
            self.learning_rate = learning_rate
            self.batch_size = batch_size

            self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_dim)
            self.layer_2 = torch.nn.Linear(self.layer_1_dim, self.layer_2_dim)
            self.layer_3 = torch.nn.Linear(self.layer_2_dim, 10)

        def train_dataloader(self):
            return DataLoader(mnist_train, batch_size=self.batch_size)

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=self.learning_rate)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--layer_1_dim', type=int, default=128)
            parser.add_argument('--layer_2_dim', type=int, default=256)
            parser.add_argument('--batch_size', type=int, default=64)
            parser.add_argument('--learning_rate', type=float, default=0.002)
            return parser

2. You can also pass in a dict or Namespace, but this obscures the parameters your module is looking
for. The user would have to search the file to find what is parametrized.

.. code-block:: python

    # using a argparse.Namespace
    class LitMNIST(LightningModule):

        def __init__(self, hparams, *args, **kwargs):
            super().__init__()
            self.hparams = hparams

            self.layer_1 = torch.nn.Linear(28 * 28, self.hparams.layer_1_dim)
            self.layer_2 = torch.nn.Linear(self.hparams.layer_1_dim, self.hparams.layer_2_dim)
            self.layer_3 = torch.nn.Linear(self.hparams.layer_2_dim, 10)

        def train_dataloader(self):
            return DataLoader(mnist_train, batch_size=self.hparams.batch_size)

One way to get around this is to convert a Namespace or dict into key-value pairs using `**`

.. code-block:: python

    parser = ArgumentParser()
    parser = LitMNIST.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    model = LitMNIST(**dict_args)

Within any LightningModule all the arguments you pass into your `__init__` will be stored in
the checkpoint so that you know all the values that went into creating this model.

We will also add all of those values to the TensorBoard hparams tab (unless it's an object which
we won't). We also will store those values into checkpoints for you which you can use to init your
models.

.. code-block:: python

    class LitMNIST(LightningModule):

        def __init__(self, layer_1_dim, some_other_param):
            super().__init__()
            self.layer_1_dim = layer_1_dim
            self.some_other_param = some_other_param

            self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_dim)

            self.layer_2 = torch.nn.Linear(self.layer_1_dim, self.some_other_param)
            self.layer_3 = torch.nn.Linear(self.some_other_param, 10)


    model = LitMNIST(10, 20)


Trainer args
^^^^^^^^^^^^
To recap, add ALL possible trainer flags to the argparser and init the Trainer this way

.. code-block:: python

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    trainer = Trainer.from_argparse_args(hparams)

    # or if you need to pass in callbacks
    trainer = Trainer.from_argparse_args(hparams, checkpoint_callback=..., callbacks=[...])


Multiple Lightning Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

We often have multiple Lightning Modules where each one has different arguments. Instead of
polluting the main.py file, the LightningModule lets you define arguments for each one.

.. testcode::

    class LitMNIST(LightningModule):

        def __init__(self, layer_1_dim, **kwargs):
            super().__init__()
            self.layer_1 = torch.nn.Linear(28 * 28, layer_1_dim)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--layer_1_dim', type=int, default=128)
            return parser

.. testcode::

    class GoodGAN(LightningModule):

        def __init__(self, encoder_layers, **kwargs):
            super().__init__()
            self.encoder = Encoder(layers=encoder_layers)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--encoder_layers', type=int, default=12)
            return parser


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

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lightning is fully compatible with the hyperparameter optimization libraries!
Here are some useful ones:

- `Hydra <https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710>`_
- `Optuna <https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py>`_
