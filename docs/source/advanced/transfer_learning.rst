.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule

================================
Transfer Learning and Finetuning
================================

Transfer learning is a training methodology where an existing pre-trained model developed for another task is used as a backbone or a starting point
to train models for new tasks. It is expected that since the existing pre-trained model is already a SOTA model, it can be used to generate better
representations of the new data, given that the pre-trained model used was trained under similar domain. This technique can help generate better models
with small dataset and models can generalize better or lead better results in less epochs as compared to training one from scratch.

Finetuning is a technique usually done after transfer learning where after pre-training, the backbone model is unfronzen and trained for a few more epochs
to improve the metric and make it more compatible with the data and current tasks, but this is not always true as sometimes it can lead to worse convergence.

*****************
Transfer Learning
*****************

Sometimes we want to use a LightningModule as a pretrained model. This is fine because
a LightningModule is just a :class:`~torch.nn.Module` but with more capabilities!

Let's use the ``AutoEncoder`` as a feature extractor in a separate model.

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

Example: Imagenet (Computer Vision)
===================================

.. testcode::
    :skipif: not _TORCHVISION_AVAILABLE

    import torchvision.models as models


    class ImagenetTransferLearning(LightningModule):
        def __init__(self):
            super().__init__()

            # init a pretrained resnet
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)

            # use the pretrained model to classify cifar-10 (10 image classes)
            num_target_classes = 10
            self.classifier = nn.Linear(num_filters, num_target_classes)

        def forward(self, x):
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
            ...

Transfer Learning...

.. code-block:: python

    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)

And use it to predict your data of interest

.. code-block:: python

    model = ImagenetTransferLearning.load_from_checkpoint(PATH)
    model.freeze()

    x = some_images_from_cifar10()
    x = images_to_tensor()
    predictions = model(x)

We used a pretrained model on imagenet, finetuned on CIFAR-10 to predict on CIFAR-10.
In the non-academic world we would finetune on a tiny dataset you have and predict on your dataset.

Example: BERT (NLP)
===================

Lightning is completely agnostic to what's used for transfer learning so long
as it is a :class:`~torch.nn.Module` subclass.

Here's a model that uses `Huggingface transformers <https://github.com/huggingface/transformers>`_.

.. testcode::

    class BertMNLIFinetuner(LightningModule):
        def __init__(self):
            super().__init__()

            self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
            self.W = nn.Linear(bert.config.hidden_size, 3)
            self.num_classes = 3

        def forward(self, input_ids, attention_mask, token_type_ids):

            h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            h_cls = h[:, 0]
            logits = self.W(h_cls)
            return logits, attn

----------

**********
Finetuning
**********

For finetuning, Lightning provides a :class:`~pytorch_lightning.callbacks.finetuning.BackboneFinetuning` callback that can integrate the finetune strategy
to train your model with a backbone. It also have :class:`~pytorch_lightning.callbacks.finetuning.BaseFinetuning` that you can subclass and add your custom
finetuning strategies as per your use-case.

.. code-block:: python

        class MyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.feature_extractor = ...
                self.linear = ...

                for p in self.feature_extractor.parameters():
                    p.requires_grad = False

            def configure_optimizer(self):
                # Make sure to filter the parameters based on `requires_grad`
                return Adam(filter(lambda p: p.requires_grad, self.parameters()))


        class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
            def __init__(self, unfreeze_at_epoch=10):
                self._unfreeze_at_epoch = unfreeze_at_epoch

            def freeze_before_training(self, pl_module):
                # freeze any module you want
                # Here, we are freezing `feature_extractor`
                self.freeze(pl_module.feature_extractor)

            def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
                # When `current_epoch` is 10, feature_extractor will start training.
                if current_epoch == self._unfreeze_at_epoch:
                    self.unfreeze_and_add_param_group(
                        modules=pl_module.feature_extractor,
                        optimizer=optimizer,
                        train_bn=True,
                    )


        finetune_strategy = FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=7)
        model = MyModel()
        trainer = Trainer(callbacks=finetine_strategy)
        trainer.fit(model)
