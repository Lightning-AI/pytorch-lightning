# Multi-node grid search gpu template  
Use this template to run a grid search on a cluster.  

## Option 1: Run on cluster using your own SLURM script    
The trainer and model will work on a cluster if you configure your SLURM script correctly.   

1. Update [this demo slurm script]().  
2. Submit the script   
```bash
$ squeue demo_script.sh
```

Most people have some way they automatically generate their own scripts.  
To run a grid search this way, you'd need a way to automatically generate scripts using all the combinations of 
hyperparameters to search over.   

## Option 2: Use test-tube for SLURM script
With test tube we can automatically generate slurm scripts for different hyperparameter options.   

Run this demo to see:  
```python
source activate YourCondaEnv

python 
```
Now you can submit a SLURM script that has the following flags   
```bash
# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:8
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=02:00:00

# activate conda env
conda activate my_env  

# run script from above
python my_test_script_above.py
```