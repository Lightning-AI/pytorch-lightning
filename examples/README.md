# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

*Note that some examples may rely on new features that are only available in the development branch and may be incompatible with any releases.*
*If you see any errors, you might want to consider switching to a version tag you would like to run examples with.*
*For example, if you're using `pytorch-lightning==1.6.4` in your environment and seeing issues, run examples of the tag [1.6.4](https://github.com/Lightning-AI/lightning/tree/1.6.4/pl_examples).*

______________________________________________________________________

## Lightning Lite Examples

We show how to accelerate your PyTorch code with [Lightning Lite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html) with minimal code changes.
You stay in full control of the training loop.

- [MNIST with vanilla PyTorch](lite/image_classifier_1_pytorch.py)
- [MNIST with Lightning Lite](lite/image_classifier_2_lite.py)

______________________________________________________________________

## Lightning Trainer Examples

In this folder, we have 2 simple examples that showcase the power of the Lightning Trainer.

- [Image Classifier](pl_basics/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Image Classifier + DALI](convert_from_pt_to_pl/image_classifier_4_dali.py) (defines the model inside the `LightningModule`).
- [Autoencoder](pl_basics/autoencoder.py)

______________________________________________________________________

## Domain Examples

This folder contains older examples. You should instead use the examples
in [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.

______________________________________________________________________

## Basic Examples

In this folder, we have 1 simple example:

- [Image Classifier + DALI](pl_integrations/dali_image_classifier.py) (defines the model inside the `LightningModule`).

______________________________________________________________________

## Loop examples

Contains implementations leveraging [loop customization](https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html) to enhance the Trainer with new optimization routines.

- [K-fold Cross Validation Loop](pl_loops/kfold.py): Implementation of cross validation in a loop and special datamodule.
- [Yield Loop](pl_loops/yielding_training_step.py): Enables yielding from the training_step like in a Python generator. Useful for automatic optimization with multiple optimizers.
