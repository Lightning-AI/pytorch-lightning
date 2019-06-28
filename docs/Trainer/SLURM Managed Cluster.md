Lightning supports model training on a cluster managed by SLURM in the following cases:    

1. Training on single or multi-cpus only.
2. Training on single or multi-gpus on the same node.
3. Coming SOON: Training across multiple nodes.

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

(3). Give trainer the cluster_manager in your main function:    

```{.python}
from pytorch_lightning import Trainer

def train_fx(trial_hparams, cluster_manager, _):
    # hparams has a specific set of hyperparams
    
    my_model = MyLightningModel()
    
    # give the trainer the cluster object
    trainer = Trainer(cluster=cluster_manager)
    trainer.fit(my_model)

```

(4). Start the grid search     
```{.python}
# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(
    train_fx, 
    nb_trials=20, 
    job_name='my_grid_search_exp_name', 
    job_display_name='my_exp')
```

That's it! The SlurmCluster object will automatically checkpoint the lightning model and resubmit if it runs into the walltime!


---
#### Walltime auto-resubmit
Lightning automatically resubmits jobs when they reach the walltime. You get this behavior for free if you give lightning
a slurm cluster object.

```{.python}
def my_main_fx(hparams, slurm_manager, _):
    trainer = Trainer(cluster=slurm_manager)
``` 

(See the grid search example above for cluster configuration).
With this feature lightning will:    

1. automatically checkpoint the model
2. checkpoint the trainer session
3. resubmit a continuation job.
4. load the checkpoint and trainer session in the new model

