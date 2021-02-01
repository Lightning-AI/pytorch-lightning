## Basic Examples
Use these examples to test how lightning works.

#### MNIST
Trains MNIST where the model is defined inside the LightningModule.
```bash
# cpu
python mnist.py

# gpus (any number)
python mnist.py

# dataparallel
python mnist.py --gpus 2 --distributed_backend 'dp'
```

---
#### MNIST with DALI
The MNIST example above using [NVIDIA DALI](https://developer.nvidia.com/DALI).
Requires NVIDIA DALI to be installed based on your CUDA version, see [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).
```bash
python mnist_dali.py
```

---
#### Image classifier
Generic image classifier with an arbitrary backbone (ie: a simple system)
```bash
# cpu
python image_classifier.py

# gpus (any number)
python image_classifier.py --gpus 2

# dataparallel
python image_classifier.py --gpus 2 --distributed_backend 'dp'
```

---
#### Autoencoder
Showing the power of a system... arbitrarily complex training loops
```bash
# cpu
python autoencoder.py

# gpus (any number)
python autoencoder.py --gpus 2

# dataparallel
python autoencoder.py --gpus 2 --distributed_backend 'dp'
```
---
# Multi-node example

This demo launches a job using 2 GPUs on 2 different nodes (4 GPUs total).
To run this demo do the following:

1. Log into the jumphost node of your SLURM-managed cluster.
2. Create a conda environment with Lightning and a GPU PyTorch version.
3. Choose a script to submit

#### DDP
Submit this job to run with DistributedDataParallel (2 nodes, 2 gpus each)
```bash
sbatch submit_ddp_job.sh YourEnv
```

#### DDP2
Submit this job to run with a different implementation of DistributedDataParallel.
In this version, each node acts like DataParallel but syncs across nodes like DDP.
```bash
sbatch submit_ddp2_job.sh YourEnv
```
