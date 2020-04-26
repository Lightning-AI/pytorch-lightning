## Basic Examples   
Use these examples to test how lightning works.   

#### Test on CPU  
```bash
python cpu_template.py
```

---   
#### Train on a single GPU
```bash
python gpu_template.py --gpus 1
```   

---    
#### DataParallel (dp)   
Train on multiple GPUs using DataParallel.

```bash
python gpu_template.py --gpus 2 --distributed_backend dp
```   

---
#### DistributedDataParallel (ddp)    

Train on multiple GPUs using DistributedDataParallel   
```bash
python gpu_template.py --gpus 2 --distributed_backend ddp
```

---
#### DistributedDataParallel+DP (ddp2)    

Train on multiple GPUs using DistributedDataParallel + DataParallel.
On a single node, uses all GPUs for 1 model. Then shares gradient information
across nodes.   
```bash
python gpu_template.py --gpus 2 --distributed_backend ddp2
```


# Multi-node example   

This demo launches a job using 2 GPUs on 2 different nodes (4 GPUs total).
To run this demo do the following:

1. Log into the jumphost node of your SLURM-managed cluster.  
2. Create a conda environment with Lightning and a GPU PyTorch version.   
3. Choose a script to submit    

#### DDP  
Submit this job to run with DistributedDataParallel (2 nodes, 2 gpus each)
```bash
sbatch ddp_job_submit.sh YourEnv
```

#### DDP2  
Submit this job to run with a different implementation of DistributedDataParallel.
In this version, each node acts like DataParallel but syncs across nodes like DDP.
```bash
sbatch ddp2_job_submit.sh YourEnv
```
