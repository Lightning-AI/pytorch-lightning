# Multi-node example   

To run this demo which launches a single job that trains on 2 nodes (2 gpus per node), do the following:

1. Log into the jumphost node of your SLURM-managed cluster.  
2. Create a conda environment with Lightning and a GPU PyTorch version.   
3. Choose a script to submit    

#### DDP  
Submit this job to run with distributedDataParallel (2 nodes, 2 gpus each)
```bash
sbatch ddp_job_submit.sh YourEnv
```

#### DDP2  
Submit this job to run with a different implementation of distributedDataParallel.
In this version, each node acts like DataParallel but syncs across nodes like DDP.
```bash
sbatch ddp2_job_submit.sh YourEnv
```
