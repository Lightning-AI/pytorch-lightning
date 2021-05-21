## Basic Examples
Use these examples to test how lightning works.

#### MNIST
Trains MNIST where the model is defined inside the `LightningModule`.
```bash
# cpu
python simple_image_classifier.py

# gpus (any number)
python simple_image_classifier.py --trainer.gpus 2

# dataparallel
python simple_image_classifier.py --trainer.gpus 2 --trainer.accelerator 'dp'
```

---
#### MNIST with DALI
The MNIST example above using [NVIDIA DALI](https://developer.nvidia.com/DALI).
Requires NVIDIA DALI to be installed based on your CUDA version, see [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).
```bash
python dali_image_classifier.py
```

---
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

---
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
---
# Multi-node example

This demo launches a job using 2 GPUs on 2 different nodes (4 GPUs total).
To run this demo do the following:

1. Log into the jumphost node of your SLURM-managed cluster.
2. Create a conda environment with Lightning and a GPU PyTorch version.
3. Choose a script to submit
