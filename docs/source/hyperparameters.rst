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

    class CoolMNIST(pl.LightningModule):
      def __init__(self, hparams):
        super(CoolMNIST, self).__init__()
        self.hparams = hparams

        self.layer_1 = torch.nn.Linear(28 * 28, hparams.layer_1_dim)
        self.layer_2 = torch.nn.Linear(hparams.layer_1_dim, hparams.layer_2_dim)
        self.layer_3 = torch.nn.Linear(hparams.layer_2_dim, 10)

      def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

      def train_dataloader(self):
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        return DataLoader(mnist_train, batch_size=self.hparams.batch_size)

      def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

      def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    hparams = parse_args()
    model = CoolMNIST(hparams)