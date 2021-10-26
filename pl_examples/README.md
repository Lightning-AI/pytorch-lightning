# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

## MNIST Examples

In this folder, we have 4 implementations of simple CNN trained over the MNIST dataset:

- [MNIST with raw PyTorch](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_1_pytorch.py) (defines the model inside the `LightningModule`).
- [MNIST with LightningLite](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_2_lite.py) (defines the model inside the `LightningModule`).
- [MNIST with LightningModule](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_3_lightning.py) (defines the model inside the `LightningModule`).
- [MNIST with LightningModule + DALI](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_examples/image_classifier_4_dali.py) (defines the model inside the `LightningModule`).

______________________________________________________________________

## Basic Examples

In this folder, we add 2 simple examples:

- [Image Classifier](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Autoencoder](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py) (shows how the `LightningModule` can be used as a system)

______________________________________________________________________

## Domain Examples

This folder contains older examples. You should instead use the examples
in [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.
