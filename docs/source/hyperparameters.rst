Hyperparameters
---------------
Normally, we don't hard-code the values to a model. We usually use the command line to
modify the network. The `Trainer` can add all the available options to an ArgumentParser.

.. code-block:: python

    from argparse import ArgumentParser

    parser = ArgumentParser()

    # add all options available in the trainer
    parser = Trainer.add_argparse_args(parser)

    # parametrize the network
    parser.add_argument('--layer_1_dim', type=int, default=128)
    parser.add_argument('--layer_1_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

Now we can parametrize our LightningModule.

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