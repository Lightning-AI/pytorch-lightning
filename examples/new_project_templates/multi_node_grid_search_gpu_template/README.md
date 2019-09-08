# Multi-node grid search gpu template  
Use this template to run a grid search on a cluster.  

## Run on cluster using your own SLURM script    
The trainer and model will work on a cluster if you configure your SLURM script correctly.   

Let's set up the trainer to run on 16 GPUs   
```python
def main():
    model = LightningModel()   
    trainer = Trainer(nb_gpu_nodes=2, gpus=[0, 1, 2, 3, 4, 5, 6, 7])   

if __name__ == '__main__':
    main()
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