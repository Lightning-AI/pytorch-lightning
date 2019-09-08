#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
conda activate my_env

# run script from above
python minimal_multi_node_demo.py