Transfer Learning
-----------------
Sometimes we want to use a LightningModule as a pretrained model. This is fine because
a LightningModule is just a `torch.nn.Module`!

Let's use the `AutoEncoder` as a feature extractor in a separate model.


.. code-block:: python

    class Encoder(torch.nn.Module):
        ...

    class AutoEncoder(pl.LightningModule):
        def __init__(self):
            self.encoder = Encoder()
            self.decoder = Decoder()

    class CIFAR10Classifier(pl.LightingModule):
        def __init__(self):
            # init the pretrained LightningModule
            self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
            self.feature_extractor.freeze()

            # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
            self.classifier = nn.Liner(100, 10)

        def forward(self, x):
            representations = self.feature_extractor(x)
            x = self.classifier(representations)
            ...

We used our pretrained Autoencoder (a LightningModule) for transfer learning!