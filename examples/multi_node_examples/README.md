# Multi-node example   

To run this demo which launches a single job that trains on 2 nodes (2 gpus per node), do the following:

1. Log into the jumphost node of your SLURM-managed cluster.  
2. Create a conda environment with Lightning and a GPU PyTorch version.   
3. Submit this script.   
```bash
sbatch job_submit.sh your_env_name_with_lightning_installed
```