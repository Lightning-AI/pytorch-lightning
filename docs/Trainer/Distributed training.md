Lightning makes multi-gpu training and 16 bit training trivial.

*Note:*   
None of the flags below require changing anything about your lightningModel definition. 

---
#### 16-bit mixed precision
16 bit precision can cut your memory footprint by half. If using volta architecture GPUs it can give a dramatic training speed-up as well.    
First, install apex (if install fails, look [here](https://github.com/NVIDIA/apex)):
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
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
# set these flags
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DEFAULT
trainer = Trainer(gpus=[0])
```

---
#### multi-gpu 
Make sure you're on a GPU machine. You can set as many GPUs as you want.
In this setting, the model will run on all 8 GPUs at once using DataParallel under the hood.
```python
# set these flags
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


trainer = Trainer(gpus=[0,1,2,3,4,5,6,7])
```

---
#### Multi-node
Multi-node training is easily done by specifying these flags.
```python
# train on 12*8 GPUs
trainer = Trainer(gpus=[0,1,2,3,4,5,6,7], nb_gpu_nodes=12)
```

In addition, make sure to set up your SLURM job correctly via the [SlurmClusterObject](https://williamfalcon.github.io/test-tube/hpc/SlurmCluster/). In particular, specify the number of tasks per node correctly.

```python
cluster = SlurmCluster(
    hyperparam_optimizer=test_tube.HyperOptArgumentParser(),
    log_path='/some/path/to/save',
)

# configure cluster
cluster.per_experiment_nb_nodes = 12 
cluster.per_experiment_nb_gpus = 8

cluster.add_slurm_cmd(cmd='ntasks-per-node', value=8, comment='1 task per gpu')
```

---
#### Self-balancing architecture
Here lightning distributes parts of your module across available GPUs to optimize for speed and memory.   

COMING SOON.
