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
            self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH).encoder
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

----

***********************************
Automated Finetuning with Callbacks
***********************************

PyTorch Lightning provides the :class:`~lightning.pytorch.callbacks.BackboneFinetuning` callback to automate
the finetuning process. This callback gradually unfreezes your model's backbone during training. This is particularly
useful when working with large pretrained models, as it allows you to start training with a frozen backbone and
then progressively unfreeze layers to fine-tune the model.

The :class:`~lightning.pytorch.callbacks.BackboneFinetuning` callback expects your model to have a specific structure:

.. testcode::

    class MyModel(LightningModule):
        def __init__(self):
            super().__init__()

            # REQUIRED: Your model must have a 'backbone' attribute
            # This should be the pretrained part you want to finetune
            self.backbone = some_pretrained_model

            # Your task-specific layers (head, classifier, etc.)
            self.head = nn.Linear(backbone_features, num_classes)

        def configure_optimizers(self):
            # Only optimize the head initially - backbone will be added automatically
            return torch.optim.Adam(self.head.parameters(), lr=1e-3)

************************************
Example: Computer Vision with ResNet
************************************

Here's a complete example showing how to use :class:`~lightning.pytorch.callbacks.BackboneFinetuning`
for computer vision:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torchvision.models as models
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import BackboneFinetuning


    class ResNetClassifier(LightningModule):
        def __init__(self, num_classes=10, learning_rate=1e-3):
            super().__init__()
            self.save_hyperparameters()

            # Create backbone from pretrained ResNet
            resnet = models.resnet50(weights="DEFAULT")
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])

            # Add custom classification head
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            # Extract features with backbone
            features = self.backbone(x)
            # Classify with head
            return self.head(features)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            # Initially only train the head - backbone will be added by callback
            return torch.optim.Adam(self.head.parameters(), lr=self.hparams.learning_rate)


    # Setup the finetuning callback
    backbone_finetuning = BackboneFinetuning(
        unfreeze_backbone_at_epoch=10,  # Start unfreezing backbone at epoch 10
        lambda_func=lambda epoch: 1.5,  # Gradually increase backbone learning rate
        backbone_initial_ratio_lr=0.1,  # Backbone starts at 10% of head learning rate
        should_align=True,  # Align rates when backbone rate reaches head rate
        verbose=True  # Print learning rates during training
    )

    model = ResNetClassifier()
    trainer = Trainer(callbacks=[backbone_finetuning], max_epochs=20)

****************************
Custom Finetuning Strategies
****************************

For more control, you can create custom finetuning strategies by subclassing
:class:`~lightning.pytorch.callbacks.BaseFinetuning`:

.. testcode::

    from lightning.pytorch.callbacks.finetuning import BaseFinetuning


    class CustomFinetuning(BaseFinetuning):
        def __init__(self, unfreeze_at_epoch=5, layers_per_epoch=2):
            super().__init__()
            self.unfreeze_at_epoch = unfreeze_at_epoch
            self.layers_per_epoch = layers_per_epoch

        def freeze_before_training(self, pl_module):
            # Freeze the entire backbone initially
            self.freeze(pl_module.backbone)

        def finetune_function(self, pl_module, epoch, optimizer):
            # Gradually unfreeze layers
            if epoch >= self.unfreeze_at_epoch:
                layers_to_unfreeze = min(
                    self.layers_per_epoch,
                    len(list(pl_module.backbone.children()))
                )

                # Unfreeze from the top layers down
                backbone_children = list(pl_module.backbone.children())
                for layer in backbone_children[-layers_to_unfreeze:]:
                    self.unfreeze_and_add_param_group(
                        layer, optimizer, lr=1e-4
                    )
