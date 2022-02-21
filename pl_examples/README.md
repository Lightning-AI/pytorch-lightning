# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

## MNIST Examples

5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html) from pure PyTorch is optional but it might be helpful to learn about it.

- [MNIST with vanilla PyTorch](./basic_examples/mnist_examples/image_classifier_1_pytorch.py)
- [MNIST with LightningLite](./basic_examples/mnist_examples/image_classifier_2_lite.py)
- [MNIST LightningLite to LightningModule](./basic_examples/mnist_examples/image_classifier_3_lite_to_lightning_module.py)
- [MNIST with LightningModule](./basic_examples/mnist_examples/image_classifier_4_lightning_module.py)
- [MNIST with LightningModule + LightningDataModule](./basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py)

______________________________________________________________________

## Basic Examples

In this folder, we have 2 simple examples:

- [Image Classifier](./basic_examples/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Image Classifier + DALI](./basic_examples/mnist_examples/image_classifier_4_dali.py) (defines the model inside the `LightningModule`).
- [Autoencoder](./basic_examples/autoencoder.py)

______________________________________________________________________

## Domain Examples

This folder contains older examples. You should instead use the examples
in [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.

______________________________________________________________________

## Basic Examples

In this folder, we have 1 simple example:

- [Image Classifier + DALI](./integration_examples/dali_image_classifier.py) (defines the model inside the `LightningModule`).

______________________________________________________________________

## Loop examples

Contains implementations leveraging [loop customization](https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html) to enhance the Trainer with new optimization routines.

- [K-fold Cross Validation Loop](./loop_examples/kfold.py): Implementation of cross validation in a loop and special datamodule.
- [Yield Loop](./loop_examples/yielding_training_step.py): Enables yielding from the training_step like in a Python generator. Useful for automatic optimization with multiple optimizers.
