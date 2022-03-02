Lightning Flash
===============

`Lightning Flash <https://lightning-flash.readthedocs.io/en/stable/>`_ is a high-level deep learning framework for fast prototyping, baselining, fine-tuning, and solving deep learning problems.
Flash makes complex AI recipes for over 15 tasks across 7 data domains accessible to all.
It is built for beginners with a simple API that requires very little deep learning background, and for data scientists, Kagglers, applied ML practitioners, and deep learning researchers that
want a quick way to get a deep learning baseline with advanced features PyTorch Lightning offers.

.. code-block:: bash

    pip install lightning-flash

-----------------

*********************************
Using Lightning Flash in 3 Steps!
*********************************

1. Load your Data
-----------------

All data loading in Flash is performed via a ``from_*`` classmethod of a ``DataModule``.
Which ``DataModule`` to use and which ``from_*`` methods are available depends on the task you want to perform.
For example, for image segmentation where your data is stored in folders, you would use the ``SemanticSegmentationData``'s `from_folders <https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html#from-folders>`_ method:

.. code-block:: python

    from flash.image import SemanticSegmentationData

    dm = SemanticSegmentationData.from_folders(
        train_folder="data/CameraRGB",
        train_target_folder="data/CameraSeg",
        val_split=0.1,
        image_size=(256, 256),
        num_classes=21,
    )

------------

2. Configure your Model
-----------------------

Our tasks come loaded with pre-trained backbones and (where applicable) heads.
You can view the available backbones to use with your task using `available_backbones <https://lightning-flash.readthedocs.io/en/latest/general/backbones.html>`_.
Once you've chosen, create the model:

.. code-block:: python

    from flash.image import SemanticSegmentation

    print(SemanticSegmentation.available_heads())
    # ['deeplabv3', 'deeplabv3plus', 'fpn', ..., 'unetplusplus']

    print(SemanticSegmentation.available_backbones("fpn"))
    # ['densenet121', ..., 'xception'] # + 113 models

    print(SemanticSegmentation.available_pretrained_weights("efficientnet-b0"))
    # ['imagenet', 'advprop']

    model = SemanticSegmentation(head="fpn", backbone="efficientnet-b0", pretrained="advprop", num_classes=dm.num_classes)

------------

3. Finetune!
------------

.. code-block:: python

    from flash import Trainer

    trainer = Trainer(max_epochs=3)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    trainer.save_checkpoint("semantic_segmentation_model.pt")


To learn more about Lightning Flash, please refer to the `Lightning Flash documentation <https://lightning-flash.readthedocs.io/en/latest/>`_.
