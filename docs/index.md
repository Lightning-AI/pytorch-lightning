###### New project Quick Start    
To start a new project you define two files, a LightningModule and a Trainer file.    

A separate trainer file allows to run many LightningModules. Each LightningModule has the core
logic to a particular research project.    

For example, one lightningModule could be an image classifier, the other
one could be a seq-2-seq model, both (optionally) ran by the same trainer file.

1. [MNIST LightningModule](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#minimal-example) 
2. [Trainer](https://williamfalcon.github.io/pytorch-lightning/Trainer/)
    - [Basic CPU Trainer Template](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/examples/new_project_templates/single_cpu_template.py) 
    - [Multi-GPU Trainer Template](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/examples/new_project_templates/single_gpu_node_template.py)
    - [GPU cluster Trainer Template](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/examples/new_project_templates/multi_node_cluster_template.py)

###### Docs shortcuts
- [LightningModule](LightningModule/RequiredTrainerInterface/)  
- [Trainer](Trainer/)  

###### Quick start examples 
- [CPU example](examples/Examples/#cpu-hyperparameter-search)   
- [Hyperparameter search on single GPU](examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)    
- [Hyperparameter search on multiple GPUs on same node](examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)  
- [Hyperparameter search on a SLURM HPC cluster](examples/Examples/#Hyperparameter search on a SLURM HPC cluster)      


###### Checkpointing    

- [Model saving](https://williamfalcon.github.io/pytorch-lightning/Trainer/Checkpointing/#model-saving)
- [Model loading](https://williamfalcon.github.io/pytorch-lightning/LightningModule/methods/#load-from-metrics) 

###### Computing cluster (SLURM)    

- [Running grid search on a cluster](https://williamfalcon.github.io/pytorch-lightning/Trainer/SLURM%20Managed%20Cluster#running-grid-search-on-a-cluster) 
- [Walltime auto-resubmit](https://williamfalcon.github.io/pytorch-lightning/Trainer/SLURM%20Managed%20Cluster#walltime-auto-resubmit)   

###### Debugging  

- [Fast dev run](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#fast-dev-run)
- [Inspect gradient norms](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#inspect-gradient-norms)
- [Log GPU usage](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#Log-gpu-usage)
- [Make model overfit on subset of data](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#make-model-overfit-on-subset-of-data)
- [Print the parameter count by layer](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#print-the-parameter-count-by-layer)
- [Pring which gradients are nan](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#print-which-gradients-are-nan)


###### Distributed training    

- [16-bit mixed precision](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#16-bit-mixed-precision)
- [Multi-GPU](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#Multi-GPU)
- [Multi-node](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#Multi-node)
- [Single GPU](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#single-gpu)
- [Self-balancing architecture](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#self-balancing-architecture)


###### Experiment Logging   

- [Display metrics in progress bar](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#display-metrics-in-progress-bar)
- [Log metric row every k batches](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#log-metric-row-every-k-batches)
- [Process position](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#process-position)
- [Tensorboard support](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#tensorboard-support)
- [Save a snapshot of all hyperparameters](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#save-a-snapshot-of-all-hyperparameters) 
- [Snapshot code for a training run](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#snapshot-code-for-a-training-run) 
- [Write logs file to csv every k batches](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#write-logs-file-to-csv-every-k-batches)

###### Training loop    

- [Accumulate gradients](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#accumulated-gradients)
- [Force training for min or max epochs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-training-for-min-or-max-epochs)
- [Force disable early stop](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-disable-early-stop)
- [Gradient Clipping](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#gradient-clipping)
- [Hooks](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/)
- [Learning rate scheduling](https://williamfalcon.github.io/pytorch-lightning/Pytorch-Lightning/LightningModule/#configure_optimizers)
- [Use multiple optimizers (like GANs)](https://williamfalcon.github.io/pytorch-lightning/Pytorch-Lightning/LightningModule/#configure_optimizers)
- [Set how much of the training set to check (1-100%)](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#set-how-much-of-the-training-set-to-check)

###### Validation loop    

- [Check validation every n epochs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#check-validation-every-n-epochs)
- [Hooks](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/)
- [Set how much of the validation set to check](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-the-number-of-validation-sanity-steps)

