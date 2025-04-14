#################
Transfer Learning
#################
**Audience**: Users looking to use pretrained models with Lightning.

----

*************************
Use any PyTorch nn.Module
*************************
Any model that is a PyTorch nn.Module can be used with Lightning (because LightningModules are nn.Modules also).

----

********************************
Use a pretrained LightningModule
********************************
Let's use the `AutoEncoder` as a feature extractor in a separate model.

.. testcode::

    class Encoder(torch.nn.Module):
        ...


    class AutoEncoder(LightningModule):
        def __init__(self):
            self.encoder = Encoder()
            self.decoder = Decoder()


    class CIFAR10Classifier(LightningModule):
        def __init__(self):
            # init the pretrained LightningModule
            self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
            self.feature_extractor.freeze()

            # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
            self.classifier = nn.Linear(100, 10)

        def forward(self, x):
            representations = self.feature_extractor(x)
            x = self.classifier(representations)
            ...

We used our pretrained Autoencoder (a LightningModule) for transfer learning!

----

***********************************
Example: Imagenet (Computer Vision)
***********************************

.. testcode::
    :skipif: not _TORCHVISION_AVAILABLE

    import torchvision.models as models


    class ImagenetTransferLearning(LightningModule):
        def __init__(self):
            super().__init__()

            # init a pretrained resnet
            backbone = models.resnet50(weights="DEFAULT")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_extractor.eval()

            # use the pretrained model to classify cifar-10 (10 image classes)
            num_target_classes = 10
            self.classifier = nn.Linear(num_filters, num_target_classes)

        def forward(self, x):
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
            ...

Finetune

.. code-block:: python

    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)

And use it to predict your data of interest

.. code-block:: python

    model = ImagenetTransferLearning.load_from_checkpoint(PATH)
    model.freeze()

    x = some_images_from_cifar10()
    predictions = model(x)

We used a pretrained model on imagenet, finetuned on CIFAR-10 to predict on CIFAR-10.
In the non-academic world we would finetune on a tiny dataset you have and predict on your dataset.

----

*******************
Example: BERT (NLP)
*******************
Lightning is completely agnostic to what's used for transfer learning so long
as it is a `torch.nn.Module` subclass.

Here's a model that uses `Huggingface transformers <https://github.com/huggingface/transformers>`_.

.. testcode::

    class BertMNLIFinetuner(LightningModule):
        def __init__(self):
            super().__init__()

            self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
            self.bert.train()
            self.W = nn.Linear(bert.config.hidden_size, 3)
            self.num_classes = 3

        def forward(self, input_ids, attention_mask, token_type_ids):
            h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            h_cls = h[:, 0]
            logits = self.W(h_cls)
            return logits, attn
