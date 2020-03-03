Hyperparameters
---------------

LightiningModule hparams
^^^^^^^^^^^^^^^^^^^^^^^^

Normally, we don't hard-code the values to a model. We usually use the command line to
modify the network. The `Trainer` can add all the available options to an ArgumentParser.

.. code-block:: python

    from argparse import ArgumentParser

    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--layer_1_dim', type=int, default=128)
    parser.add_argument('--layer_1_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

Now we can parametrize the LightningModule.

.. code-block:: python
    :emphasize-lines: 5,6,7,12,14

    class CoolMNIST(pl.LightningModule):
      def __init__(self, hparams):
        super(CoolMNIST, self).__init__()
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
    model = CoolMNIST(hparams)

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
        model = CoolMNIST(hparams=args)
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

    python main.py --layer_1_dim 128 --layer_2_dim 256 --batch_size 64 --max_epochs 64

But it would also be nice to pass in any arbitrary argument to the trainer.
We can do it by changing how we init the trainer.

.. code-block:: python

    def main(args):
        model = CoolMNIST(hparams=args)

        # makes all trainer options available from the command line
        trainer = Trainer.from_argparse_args(args)

and now we can do this:

.. code-block:: bash

    python main.py --gpus 1 --min_epochs 12 --max_epochs 64 --arbitrary_trainer_arg some_value



