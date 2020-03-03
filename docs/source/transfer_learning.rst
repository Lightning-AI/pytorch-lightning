Transfer Learning
-----------------

Using Pretrained Models
^^^^^^^^^^^^^^^^^^^^^^^

Sometimes we want to use a LightningModule as a pretrained model. This is fine because
a LightningModule is just a `torch.nn.Module`!

.. note:: Remember that a pl.LightningModule is EXACTLY a torch.nn.Module but with more capabilities.

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
            self.classifier = nn.Linear(100, 10)

        def forward(self, x):
            representations = self.feature_extractor(x)
            x = self.classifier(representations)
            ...

We used our pretrained Autoencoder (a LightningModule) for transfer learning!

Example: BERT (transformers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lightning is completely agnostic to what's used for transfer learning so long
as it is a `torch.nn.Module` subclass.

.. code-block:: python

    from transformers import BertModel

    class BertMNLIFinetuner(pl.LightningModule):

    def __init__(self):
        super(BertMNLIFinetuner, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3


    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn