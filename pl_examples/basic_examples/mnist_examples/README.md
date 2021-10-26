## MNIST Examples

This tutorial contains 5 examples implementing simple ImageClassifier trained over MNIST.
They demonstrate how to slowly convert from raw PyTorch to PyTorch Lightning.

#### 1 . Image Classifier with PyTorch

Trains a simple CNN over MNIST using raw PyTorch.

```bash
# cpu
python image_classifier_1_pytorch.py
```

______________________________________________________________________

#### 2. Image Classifier with LightningLite

Trains a simple CNN over MNIST using [LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.rst).

```bash
# cpu / multiple gpus if available
python image_classifier_2_lite.py
```

______________________________________________________________________

#### 3. Image Classifier - Conversion Lite to Lightning.

Trains MNIST where the model is defined inside the `LightningModule`.

```bash
# cpu
python image_classifier_3_lite_to_lightning.py

# gpus (any number)
python image_classifier_3_lite_to_lightning.py --trainer.gpus 2

# dataparallel
python image_classifier_3_lite_to_lightning.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

#### 4. Image Classifier with Lightning.

Trains MNIST where the model is defined inside the `LightningModule`.

```bash
# cpu
python image_classifier_4_lightning.py

# gpus (any number)
python image_classifier_4_lightning.py --trainer.gpus 2

# dataparallel
python image_classifier_4_lightning.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

______________________________________________________________________

#### 5. Image Classifier with DALI

The MNIST example above using [NVIDIA DALI](https://developer.nvidia.com/DALI).
Requires NVIDIA DALI to be installed based on your CUDA version, see [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).

```bash
python image_classifier_5_dali.py
```

______________________________________________________________________

# Multi-node example

This demo launches a job using 2 GPUs on 2 different nodes (4 GPUs total).
To run this demo do the following:

1. Log into the jumphost node of your SLURM-managed cluster.
1. Create a conda environment with Lightning and a GPU PyTorch version.
1. Choose a script to submit
