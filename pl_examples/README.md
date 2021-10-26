# Examples

Our most robust examples showing all sorts of implementations
can be found in our sister library [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html).

______________________________________________________________________

## Basic examples

In this folder we add several starter examples:

- [MNIST Classifier](./basic_examples/simple_image_classifier.py): Shows how to define the model inside the `LightningModule`.
- [Image Classifier](./basic_examples/backbone_image_classifier.py): Trains arbitrary datasets with arbitrary backbones.
- [Autoencoder](./basic_examples/autoencoder.py): Shows how the `LightningModule` can be used as a system.
- [Profiler](./basic_examples/profiler_example.py): Shows the basic usage of the PyTorch profilers and how to inspect traces in Google Chrome.
- [Image Classifier with DALI](./basic_examples/dali_image_classifier.py): Shows how to use [NVIDIA DALI](https://developer.nvidia.com/DALI) with Lightning.
- [Mnist Datamodule](.basic_examples/mnist_datamodule.py): Shows how to define a simple `LightningDataModule` using the MNIST dataset.

______________________________________________________________________

## Domain examples

This folder contains older examples. You should instead use the examples
in [Lightning Bolts](https://pytorch-lightning.readthedocs.io/en/latest/ecosystem/bolts.html)
for advanced use cases.

______________________________________________________________________

## Loop examples

Contains implementations leveraging [loop customization](https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html) to enhance the Trainer with new optimization routines.

- [K-fold Cross Validation Loop](./loop_examples/kfold.py): Implemenation of cross validation in a loop and special datamodule.
- [Yield Loop](./loop_examples/yielding_training_step.py): Enables yielding from the training_step like in a Python generator. Useful for automatic optimization with multiple optimizers.
