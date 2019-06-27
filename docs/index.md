# PYTORCH-LIGHTNING DOCUMENTATION

###### Quick start
- Define a lightning model   
- Set up the trainer   

###### Quick start examples 
- CPU example   
- Single GPU example   
- Multi-gpu example 
- SLURM cluster example      


###### Distributed training
- Single-gpu      
- Multi-gpu      
- Multi-node   

###### Checkpointing
- Model saving
- Model loading 

###### Computing cluster (SLURM)
- Automatic checkpointing   
- Automatic saving, loading   
- Walltime auto-resubmit   

###### Common training use cases 
- 16-bit mixed precision
- Accumulate gradients
- Check val many times during 1 training epoch
- Check GPU usage
- Check validation every n epochs
- Check which gradients are nan
- Inspect gradient norms
- Learning rate annealing
- Make model overfit on subset of data
- Min, max epochs
- Multiple optimizers (like GANs)
- Run a sanity check of model val and tng step
- Set how much of the tng, val, test sets to check (1-100%)
