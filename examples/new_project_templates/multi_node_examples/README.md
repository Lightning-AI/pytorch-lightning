# Multi-node examples
Use these templates for multi-node training. 
The main complexity around cluster training is how you submit the SLURM jobs.  

## Test-tube   
Lightning uses test-tube to submit SLURM jobs and to run hyperparameter searches on a cluster.  

To run a hyperparameter search, we normally add the values to search to the Hyperparameter optimizer 
```python
from test_tube import HyperOptArgumentParser

parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=True)
parser.opt_list('--learning_rate', default=0.001, type=float,
                        options=[0.0001, 0.0005, 0.001],
                        tunable=True)
                        
# give your model a chance to add its own parameters
parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)

# parse args
hyperparams = parser.parse_args()
```

The above sets up a grid search on learning rate and drop probability. You can now add this object to the 
cluster object to perform the grid search:   
```python
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/path/to/log/slurm/files',
)

# ... configure cluster options

# run grid search on cluster
nb_trials = 6   # (2 drop probs * 3 lrs)
cluster.optimize_parallel_cluster_gpu(
    YourMainFunction,
    nb_trials=nb_trials,
    job_name=hyperparams.experiment_name
)
```

Running the above will launch 6 jobs, each with a different drop prob and learning rate combination.   
The ```tunable``` parameter must be set to True to add that argument to the space of options, otherwise
Test-Tube will use the ```default=value```.    


## SLURM Flags   
However you decide to submit your jobs, debugging requires a few flags. Without these flags, you'll
see a nccl error instead of the actual error which caused the bug.   

```sh
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
```

On some clusters you might need to set the network interface with this flag.   
```sh
export NCCL_SOCKET_IFNAME=^docker0,lo
```   

You might also need to load the latest version of NCCL  
```sh
module load NCCL/2.4.7-1-cuda.10.0
```

Finally, you must set the master port (usually a random number between 12k and 20k).   
```sh
# random port between 12k and 20k
export MASTER_PORT=$((12000 + RANDOM % 20000))$   
```

## Simplest example.   
1. Modify this script with your CoolModel file.   
2. Update and submit [this bash script](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/multi_node_examples/minimal_multi_node_demo_script.sh)   
```bash
squeue minimal_multi_node_demo_script.sh
```

## Grid search on a cluster   

#### Option 1: Run on cluster using your own SLURM script    
The trainer and model will work on a cluster if you configure your SLURM script correctly.   

1. Update [this demo slurm script](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/multi_node_examples/demo_script.sh).  
2. Submit the script   
```bash
$ squeue demo_script.sh
```

Most people have some way they automatically generate their own scripts.  
To run a grid search this way, you'd need a way to automatically generate scripts using all the combinations of 
hyperparameters to search over.   

#### Option 2: Use test-tube for SLURM script
With test tube we can automatically generate slurm scripts for different hyperparameter options.   

To run this demo:    
```bash
source activate YourCondaEnv

python multi_node_cluster_auto_slurm.py --email your@email.com --gpu_partition your_partition --conda_env YourCondaEnv
```

That will submit 6 jobs. Each job will have a specific combination of hyperparams. Each job will also run on 2 nodes
where each node has 8 gpus.   
