Lightning makes multi-gpu training and 16 bit training trivial.

*Note:*   
None of the flags below require changing anything about your lightningModel definition. 

---
#### Choosing a backend  
Lightning supports two backends. DataParallel and DistributedDataParallel. Both can be used for single-node multi-GPU training.
For multi-node training you must use DistributedDataParallel.   

##### DataParallel (dp)   
Splits a batch across multiple GPUs on the same node. Cannot be used for multi-node training.   

##### DistributedDataParallel (ddp)   
Trains a copy of the model on each GPU and only syncs gradients. If used with DistributedSampler, each GPU trains
on a subset of the full dataset.  

##### DistributedDataParallel-2 (ddp2)   
Works like DDP, except each node trains a single copy of the model using ALL GPUs on that node. 
Very useful when dealing with negative samples, etc...

You can toggle between each mode by setting this flag.
``` {.python}
# DEFAULT (when using single GPU or no GPUs)
trainer = Trainer(distributed_backend=None)

# Change to DataParallel (gpus > 1)
trainer = Trainer(distributed_backend='dp')

# change to distributed data parallel (gpus > 1)
trainer = Trainer(distributed_backend='ddp')

# change to distributed data parallel (gpus > 1)
trainer = Trainer(distributed_backend='ddp2')
```

If you request multiple nodes, the back-end will auto-switch to ddp.
We recommend you use DistributedDataparallel even for single-node multi-GPU training. It is MUCH faster than DP but *may*
have configuration issues depending on your cluster.

For a deeper understanding of what lightning is doing, feel free to read [this guide](https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565).   

---
#### Distributed and 16-bit precision.    
Due to an issue with apex and DistributedDataParallel (PyTorch and NVIDIA issue), Lightning does
not allow 16-bit and DP training. We tried to get this to work, but it's an issue on their end.   

Below are the possible configurations we support.    

| 1 GPU  | 1+ GPUs  | DP  | DDP  | 16-bit  | command |
|---|---|---|---|---|---|
| Y  |   |   |   |  | ```Trainer(gpus=1)``` |
| Y  |   |   |   | Y | ```Trainer(gpus=1, use_amp=True)``` |
|   | Y | Y |   |   | ```Trainer(gpus=k, distributed_backend='dp')``` |
|   | Y |  | Y  |  | ```Trainer(gpus=k, distributed_backend='ddp')``` |
|   | Y |  | Y  | Y | ```Trainer(gpus=k, distributed_backend='ddp', use_amp=True)``` |

You also have the option of specifying which GPUs to use by passing a list:   

```python
# DEFAULT (int)
Trainer(gpus=k)  

# You specify which GPUs (don't use if running on cluster)  
Trainer(gpus=[0, 1])  

# can also be a string
Trainer(gpus='0, 1')
```

---
#### CUDA flags   
CUDA flags make certain GPUs visible to your script. 
Lightning sets these for you automatically, there's NO NEED to do this yourself.
```python
# lightning will set according to what you give the trainer
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

However, when using a cluster, Lightning will NOT set these flags (and you should not either). 
SLURM will set these for you.   

---
#### 16-bit mixed precision
16 bit precision can cut your memory footprint by half. If using volta architecture GPUs it can give a dramatic training speed-up as well.    
First, install apex (if install fails, look [here](https://github.com/NVIDIA/apex)):
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex

# ------------------------
# OPTIONAL: on your cluster you might need to load cuda 10 or 9
# depending on how you installed PyTorch

# see available modules
module avail

# load correct cuda before install
module load cuda-10.0
# ------------------------

# make sure you've loaded a cuda version > 4.0 and < 7.0
module load gcc-6.1.0

$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

then set this use_amp to True.
``` {.python}
# DEFAULT
trainer = Trainer(amp_level='O2', use_amp=False)
```

---
#### Single-gpu
Make sure you're on a GPU machine. 
```python
# DEFAULT
trainer = Trainer(gpus=1)
```

---
#### multi-gpu 
Make sure you're on a GPU machine. You can set as many GPUs as you want.
In this setting, the model will run on all 8 GPUs at once using DataParallel under the hood.
```python
# to use DataParallel
trainer = Trainer(gpus=8, distributed_backend='dp')

# RECOMMENDED use DistributedDataParallel
trainer = Trainer(gpus=8, distributed_backend='ddp')
```

---
#### Multi-node
Multi-node training is easily done by specifying these flags. 
```python
# train on 12*8 GPUs
trainer = Trainer(gpus=8, nb_gpu_nodes=12, distributed_backend='ddp')
```

You must configure your job submission script correctly for the trainer to work. Here is an example
script for the above trainer configuration.   

```sh
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=12
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
conda activate my_env

# -------------------------
# OPTIONAL
# -------------------------
# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# PyTorch comes with prebuilt NCCL support... but if you have issues with it
# you might need to load the latest version from your  modules
# module load NCCL/2.4.7-1-cuda.10.0

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
# -------------------------

# random port between 12k and 20k
export MASTER_PORT=$((12000 + RANDOM % 20000))

# run script from above
python my_main_file.py
```

**NOTE:** When running in DDP mode, any errors in your code will show up as an NCCL issue.
Set the ```NCCL_DEBUG=INFO``` flag to see the ACTUAL error.

Finally, make sure to add a distributed sampler to your dataset. The distributed sampler copies a 
portion of your dataset onto each GPU. (World_size = gpus_per_node * nb_nodes).   

```python
# ie: this:
dataset = myDataset()
dataloader = Dataloader(dataset)

# becomes:
dataset = myDataset()
dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = Dataloader(dataset, sampler=dist_sampler)
```

#### Auto-slurm-job-submission
Instead of manually building SLURM scripts, you can use the [SlurmCluster object](https://williamfalcon.github.io/test-tube/hpc/SlurmCluster/) to
do this for you. The SlurmCluster can also run a grid search if you pass in a [HyperOptArgumentParser](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/).

Here is an example where you run a grid search of 9 combinations of hyperparams.
[The full examples are here](https://github.com/williamFalcon/pytorch-lightning/tree/master/examples/new_project_templates/multi_node_examples).
```python
# grid search 3 values of learning rate and 3 values of number of layers for your net
# this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
parser.opt_list('--learning_rate', default=0.001, type=float, options=[1e-3, 1e-2, 1e-1], tunable=True)
parser.opt_list('--layers', default=1, type=float, options=[16, 32, 64], tunable=True)
hyperparams = parser.parse_args()

# Slurm cluster submits 9 jobs, each with a set of hyperparams
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/some/path/to/save',
)

# OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
# which interface your nodes use for communication
cluster.add_command('export NCCL_SOCKET_IFNAME=^docker0,lo')

# see output of the NCCL connection process
# NCCL is how the nodes talk to each other
cluster.add_command('export NCCL_DEBUG=INFO')

# setting a master port here is a good idea.
cluster.add_command('export MASTER_PORT=%r' % PORT)

# ************** DON'T FORGET THIS ***************
# MUST load the latest NCCL version
cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

# configure cluster
cluster.per_experiment_nb_nodes = 12 
cluster.per_experiment_nb_gpus = 8

cluster.add_slurm_cmd(cmd='ntasks-per-node', value=8, comment='1 task per gpu')  

# submit a script with 9 combinations of hyper params
# (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
cluster.optimize_parallel_cluster_gpu(
    main,
    nb_trials=9, # how many permutations of the grid search to run
    job_name='name_for_squeue'
)
```

The other option is that you generate scripts on your own via a bash command or use another library...

---
#### Self-balancing architecture
Here lightning distributes parts of your module across available GPUs to optimize for speed and memory.   

COMING SOON.
