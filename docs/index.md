# PYTORCH-LIGHTNING DOCUMENTATION

###### Main Docs
- [LightningModule](Pytorch-Lightning/LightningModule)  
- [Trainer](Trainer/)  

###### New project Quick Start
- [Define a LightningModule](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/lightning_module_template.py)  

Pick a trainer
- [Basic CPU Trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/trainer_cpu_template.py) 
- [GPU cluster Trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/trainer_gpu_cluster_template.py)

###### Quick start examples 
- CPU example   
- Single GPU example   
- Multi-gpu example 
- SLURM cluster grid search example      

###### Training loop
- Accumulate gradients
- Check GPU usage
- Check which gradients are nan
- Check validation every n epochs
- Display metrics in progress bar
- Force training for min or max epochs
- Inspect gradient norms
- Hooks
- Learning rate annealing
- Make model overfit on subset of data
- Multiple optimizers (like GANs)
- Set how much of the training set to check (1-100%)
- training_step function

###### Validation loop
- Display metrics in progress bar
- hooks
- Set how much of the validation set to check (1-100%)
- Set validation check frequency within 1 training epoch (1-100%)
- validation_step function
- Why does validation run first for 5 steps?

###### Distributed training
- Single-gpu      
- Multi-gpu      
- Multi-node   
- 16-bit mixed precision

###### Checkpointing
- Model saving
- Model loading 

###### Computing cluster (SLURM)
- Automatic checkpointing   
- Automatic saving, loading  
- Running grid search on a cluster 
- Walltime auto-resubmit   
