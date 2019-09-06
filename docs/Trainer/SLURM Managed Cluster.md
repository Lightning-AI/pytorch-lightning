Lightning supports model training on a cluster managed by SLURM in the following cases:    

1. Training on a single cpu or single GPU.
2. Train on multiple GPUs on the same node using DataParallel or DistributedDataParallel
3. Training across multiple GPUs on multiple different nodes via DistributedDataParallel.

**Note: A node means a machine with multiple GPUs**

---
#### Running grid search on a cluster
To use lightning to run a hyperparameter search (grid-search or random-search) on a cluster do 4 things:   

(1). Define the parameters for the grid search    
    
```{.python}
from test_tube import HyperOptArgumentParser

# subclass of argparse
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')

# let's enable optimizing over the number of layers in the network
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])

hparams = parser.parse_args()    
```    
    
     
(2). Define the cluster options in the [SlurmCluster object](https://williamfalcon.github.io/test-tube/hpc/SlurmCluster/) (over 5 nodes and 8 gpus)    

```{.python}
from test_tube.hpc import SlurmCluster

# hyperparameters is a test-tube hyper params object
# see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
hyperparams = args.parse()

# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/path/to/log/results/to',
    python_cmd='python3'
)

# let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 8
cluster.per_experiment_nb_nodes = 5

# we'll request 10GB of memory per node
cluster.memory_mb_per_node = 10000

# set a walltime of 10 minues
cluster.job_time = '10:00'
```

(3). Make a main function with your model and trainer. Each job will call this function with a particular
hparams configuration.    
```{.python}
from pytorch_lightning import Trainer

def train_fx(trial_hparams, cluster_manager, _):
    # hparams has a specific set of hyperparams
    
    my_model = MyLightningModel()
    
    # give the trainer the cluster object
    trainer = Trainer()
    trainer.fit(my_model)

```

(3). Start the grid/random search     
```{.python}
# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(
    train_fx, 
    nb_trials=20, 
    job_name='my_grid_search_exp_name', 
    job_display_name='my_exp')
```

---
#### Walltime auto-resubmit
Lightning automatically resubmits jobs when they reach the walltime. Make sure to set the SIGUSR1 signal in 
your SLURM script.   

```bash
# 90 seconds before training ends
#SBATCH --signal=SIGUSR1@90
``` 

When lightning receives the SIGUSR1 signal it will:
1. save a checkpoint with 'hpc_ckpt' in the name.
2. resubmit the job using the SLURM_JOB_ID  

When the script starts again, Lightning will:
1. search for a 'hpc_ckpt' checkpoint. 
2. restore the model, optimizers, schedulers, epoch, etc...   


