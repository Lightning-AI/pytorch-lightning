## Basic Examples

Use these examples to test how lightning works.

## MNIST Examples

The following examples contain 5 MNIST examples showing how to gradually convert from pure PyTorch to PyTorch Lightning.

#### 1 . Image Classifier with PyTorch

Trains a simple CNN over MNIST using raw PyTorch.

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

#### 3. Image Classifier - Conversion Lite to Lightning.

Trains a simple CNN over MNIST with a `LightningModule` and `LightningLite`.

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_3_lite_to_lightning.py
```

______________________________________________________________________

#### 4. Image Classifier - Conversion Lite to Lightning + Lightning Loops

Trains a simple CNN over MNIST with a `LightningModule` and `LightningLite` and `Loops`.

```bash
# cpu / multiple gpus if available
python mnist_examples/image_classifier_4_lite_to_lightning_and_loops.py
```

______________________________________________________________________

#### 5. Image Classifier with Lightning.

Trains a simple CNN over MNIST with a `Trainer` and `LightningModule`.

```bash
# cpu
python mnist_examples/image_classifier_5_lightning.py

# gpus (any number)
python mnist_examples/image_classifier_5_lightning.py --trainer.gpus 2

# distributed data parallel
python mnist_examples/image_classifier_5_lightning.py --trainer.gpus 2 --trainer.strategy 'ddp'
```

______________________________________________________________________

#### Image classifier

Generic image classifier with an arbitrary backbone (ie: a simple system)

```bash
# cpu
python backbone_image_classifier.py

# gpus (any number)
python backbone_image_classifier.py --trainer.gpus 2

# dataparallel
python backbone_image_classifier.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

#### Image Classifier with DALI

The MNIST example above using [NVIDIA DALI](https://developer.nvidia.com/DALI).
Requires NVIDIA DALI to be installed based on your CUDA version, see [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).

```bash
python dali_image_classifier.py.py
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
