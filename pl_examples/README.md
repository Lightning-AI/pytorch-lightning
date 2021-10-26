# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

## MNIST Examples

5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst) from pure PyTorch is optional but it might helpful to learn about it.

- [MNIST with vanilla PyTorch](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_1_pytorch.py)
- [MNIST with LightningLite](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_2_lite.py)
- [MNIST LightningLite to PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_3_lite_to_lightning.py)
- [MNIST with LightningModule](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_4_lightning.py)
- [MNIST with LightningModule + LightningDataModule](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py)

______________________________________________________________________

## Basic Examples

In this folder, we add 2 simple examples:

- [Image Classifier](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Image Classifier + DALI](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_4_dali.py) (defines the model inside the `LightningModule`).
- [Autoencoder](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py) (shows how the `LightningModule` can be used as a system)

______________________________________________________________________

## Domain Examples

This folder contains older examples. You should instead use the examples
in [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.
