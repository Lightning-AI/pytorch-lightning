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

    hparams = parser.parse_args()

Now you can call run your program like so

.. code-block:: bash

    python trainer_main.py --gpus 2 --num_nodes 2 --conda_env 'my_env' --encoder_layers 12

Finally, make sure to start the training like so:

.. code-block:: python

    # YES
    model = LitModel(hparams)
    trainer = Trainer.from_argparse_args(hparams, early_stopping_callback=...)

    # NO
    # model = LitModel(learning_rate=hparams.learning_rate, ...)
    # trainer = Trainer(gpus=hparams.gpus, ...)

LightningModule hparams
^^^^^^^^^^^^^^^^^^^^^^^

Normally, we don't hard-code the values to a model. We usually use the command line to
modify the network and read those values in the LightningModule

.. testcode::

    class LitMNIST(LightningModule):

        def __init__(self, hparams):
            super().__init__()

            # do this to save all arguments in any logger (tensorboard)
            self.hparams = hparams

            self.layer_1 = torch.nn.Linear(28 * 28, hparams.layer_1_dim)
            self.layer_2 = torch.nn.Linear(hparams.layer_1_dim, hparams.layer_2_dim)
            self.layer_3 = torch.nn.Linear(hparams.layer_2_dim, 10)

        def train_dataloader(self):
            return DataLoader(mnist_train, batch_size=self.hparams.batch_size)

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=self.hparams.learning_rate)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--layer_1_dim', type=int, default=128)
            parser.add_argument('--layer_2_dim', type=int, default=256)
            parser.add_argument('--batch_size', type=int, default=64)
            parser.add_argument('--learning_rate', type=float, default=0.002)
            return parser

Now pass in the params when you init your model

.. code-block:: python

    parser = ArgumentParser()
    parser = LitMNIST.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = LitMNIST(hparams)

The line `self.hparams = hparams` is very special. This line assigns your hparams to the LightningModule.
This does two things:

1.  It adds them automatically to TensorBoard logs under the hparams tab.
2.  Lightning will save those hparams to the checkpoint and use them to restore the module correctly.

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

        def __init__(self, hparams):
            super().__init__()
            self.layer_1 = torch.nn.Linear(28 * 28, hparams.layer_1_dim)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser])
            parser.add_argument('--layer_1_dim', type=int, default=128)
            return parser

.. testcode::

    class GoodGAN(LightningModule):

        def __init__(self, hparams):
            super().__init__()
            self.encoder = Encoder(layers=hparams.encoder_layers)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser])
            parser.add_argument('--encoder_layers', type=int, default=12)
            return parser


Now we can allow each model to inject the arguments it needs in the ``main.py``

.. code-block:: python

    def main(args):

        # pick model
        if args.model_name == 'gan':
            model = GoodGAN(hparams=args)
        elif args.model_name == 'mnist':
            model = LitMNIST(hparams=args)

        model = LitMNIST(hparams=args)
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
