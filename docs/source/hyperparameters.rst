Hyperparameters
---------------
Lightning has utilities to interact seamlessly with the command line ArgumentParser
and plays well with the hyperparameter optimization framework of your choice.

LightiningModule hparams
^^^^^^^^^^^^^^^^^^^^^^^^

Normally, we don't hard-code the values to a model. We usually use the command line to
modify the network. The `Trainer` can add all the available options to an ArgumentParser.

.. code-block:: python

    from argparse import ArgumentParser

    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--layer_1_dim', type=int, default=128)
    parser.add_argument('--layer_2_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

Now we can parametrize the LightningModule.

.. code-block:: python
    :emphasize-lines: 5,6,7,12,14

    class LitMNIST(pl.LightningModule):
      def __init__(self, hparams):
        super(LitMNIST, self).__init__()
        self.hparams = hparams

        self.layer_1 = torch.nn.Linear(28 * 28, hparams.layer_1_dim)
        self.layer_2 = torch.nn.Linear(hparams.layer_1_dim, hparams.layer_2_dim)
        self.layer_3 = torch.nn.Linear(hparams.layer_2_dim, 10)

      def forward(self, x):
        ...

      def train_dataloader(self):
        ...
        return DataLoader(mnist_train, batch_size=self.hparams.batch_size)

      def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    hparams = parse_args()
    model = LitMNIST(hparams)

.. note:: Bonus! if (hparams) is in your module, Lightning will save it into the checkpoint and restore your
    model using those hparams exactly.

And we can also add all the flags available in the Trainer to the Argparser.

.. code-block:: python

    # add all the available Trainer options to the ArgParser
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

And now you can start your program with

.. code-block:: bash

    # now you can use any trainer flag
    $ python main.py --num_nodes 2 --gpus 8

Trainer args
^^^^^^^^^^^^

It also gets annoying to map each argument into the Argparser. Luckily we have
a default parser

.. code-block:: python

    parser = ArgumentParser()

    # add all options available in the trainer such as (max_epochs, etc...)
    parser = Trainer.add_argparse_args(parser)

We set up the main training entry point file like this:

.. code-block:: python

    def main(args):
        model = LitMNIST(hparams=args)
        trainer = Trainer(max_epochs=args.max_epochs)
        trainer.fit(model)

    if __name__ == '__main__':
        parser = ArgumentParser()

        # adds all the trainer options as default arguments (like max_epochs)
        parser = Trainer.add_argparse_args(parser)

        # parametrize the network
        parser.add_argument('--layer_1_dim', type=int, default=128)
        parser.add_argument('--layer_1_dim', type=int, default=256)
        parser.add_argument('--batch_size', type=int, default=64)
        args = parser.parse_args()

        # train
        main(args)

And now we can train like this:

.. code-block:: bash

    $ python main.py --layer_1_dim 128 --layer_2_dim 256 --batch_size 64 --max_epochs 64

But it would also be nice to pass in any arbitrary argument to the trainer.
We can do it by changing how we init the trainer.

.. code-block:: python

    def main(args):
        model = LitMNIST(hparams=args)

        # makes all trainer options available from the command line
        trainer = Trainer.from_argparse_args(args)

and now we can do this:

.. code-block:: bash

    $ python main.py --gpus 1 --min_epochs 12 --max_epochs 64 --arbitrary_trainer_arg some_value

Multiple Lightning Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

We often have multiple Lightning Modules where each one has different arguments. Instead of
polluting the main.py file, the LightningModule lets you define arguments for each one.

.. code-block:: python

    class LitMNIST(pl.LightningModule):
      def __init__(self, hparams):
        super(LitMNIST, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, hparams.layer_1_dim)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser])
            parser.add_argument('--layer_1_dim', type=int, default=128)
            return parser

    class GoodGAN(pl.LightningModule):
      def __init__(self, hparams):
        super(GoodGAN, self).__init__()
        self.encoder = Encoder(layers=hparams.encoder_layers)

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser])
            parser.add_argument('--encoder_layers', type=int, default=12)
            return parser

Now we can allow each model to inject the arguments it needs in the main.py

.. code-block:: python

    def main(args):

        # pick model
        if args.model_name == 'gan':
            model = GoodGAN(hparams=args)
        elif args.model_name == 'mnist':
            model = LitMNIST(hparams=args)

        model = LitMNIST(hparams=args)
        trainer = Trainer(max_epochs=args.max_epochs)
        trainer.fit(model)

    if __name__ == '__main__':
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)

        # figure out which model to use
        parser.add_argument('--model_name', type=str, default='gan', help='gan or mnist')
        temp_args = parser.parse_known_args()

        # let the model add what it wants
        if temp_args.model_name == 'gan':
            parser = GoodGAN.add_model_specific_args(parser)
        elif temp_args.model_name == 'mnist':
            parser = LitMNIST.add_model_specific_args(parser)

        args = parser.parse_args()

        # train
        main(args)

and now we can train MNIST or the gan using the command line interface!

.. code-block:: bash

    $ python main.py --model_name gan --encoder_layers 24
    $ python main.py --model_name mnist --layer_1_dim 128

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lightning is fully compatible with the hyperparameter optimization libraries!
Here are some useful ones:

- `Hydra <https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710>`_
- `Optuna <https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py>`_
