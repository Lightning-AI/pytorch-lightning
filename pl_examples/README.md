# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

## Basic examples

In this folder we add 3 simple examples:

- [MNIST Classifier](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/simple_image_classifier.py) (defines the model inside the `LightningModule`).
- [Image Classifier](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/backbone_image_classifier.py) (trains arbitrary datasets with arbitrary backbones).
- [Autoencoder](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py) (shows how the `LightningModule` can be used as a system)

______________________________________________________________________

## Domain examples

This folder contains older examples. You should instead use the examples
in [lightning-bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.

______________________________________________________________________

## Loop examples

Contains implementations leveraging [loop customization](https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html) to enhance the Trainer with new optimization routines.

- [K-fold Cross Validation Loop](./loop_examples/kfold.py): Implemenation of cross validation in a loop and special datamodule.
- [Yield Loop](./loop_examples/kfold.py): Enables yielding from the training_step like in a Python generator. Useful for automatic optimization with multiple optimizers.
