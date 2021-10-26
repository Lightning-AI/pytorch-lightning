## Basic Examples

Use these examples to test how lightning works.

## MNIST Examples

5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

The transition through [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst) from pure PyTorch is optional but it might helpful to learn about it.

#### 1 . Image Classifier with Vanilla PyTorch

Trains a simple CNN over MNIST using vanilla PyTorch.

```bash
# cpu
python mnist_examples/image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

Trains a simple CNN over MNIST using [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst).

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_2_lite.py
```

______________________________________________________________________

Trains a simple CNN over MNIST where `LightningLite` is almost a `LightningModule`.

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_3_lite_to_lightning.py
```

______________________________________________________________________

#### 4. Image Classifier with LightningModule

Trains a simple CNN over MNIST with `Lightning Trainer` and the converted `LightningModule`.

```bash
# cpu
python mnist_examples/image_classifier_4_lightning.py

# gpus (any number)
python mnist_examples/image_classifier_4_lightning.py --trainer.gpus 2
```

______________________________________________________________________

#### 5. Image Classifier with LightningModule + LightningDataModule

Trains a simple CNN over MNIST with `Lightning Trainer` and the converted `LightningModule` and `LightningDataModule`

```bash
# cpu
python mnist_examples/image_classifier_5_lightning_datamodule.py

# gpus (any number)
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.gpus 2

# data parallel
python mnist_examples/image_classifier_5_lightning_datamodule.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

#### Autoencoder

Showing the power of a system... arbitrarily complex training loops

```bash
# cpu
python autoencoder.py

# gpus (any number)
python autoencoder.py --trainer.gpus 2

# dataparallel
python autoencoder.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

# Multi-node example

This demo launches a job using 2 GPUs on 2 different nodes (4 GPUs total).
To run this demo do the following:

1. Log into the jumphost node of your SLURM-managed cluster.
1. Create a conda environment with Lightning and a GPU PyTorch version.
1. Choose a script to submit
