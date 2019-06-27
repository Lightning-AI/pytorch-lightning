# Trainer
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/models/trainer.py)]

The lightning trainer abstracts best practices for running a training, val, test routine. It calls parts of your model when it wants to hand over full control and otherwise makes training assumptions which are now standard practice in AI research.

This is the basic use of the trainer:

``` {.python}
from pytorch_lightning import Trainer

model = LightningTemplate()

trainer = Trainer()
trainer.fit(model)
```

But of course the fun is in all the advanced things it can do:

**Training loop**    

- Accumulate gradients
- Check GPU usage
- Check which gradients are nan
- Check validation every n epochs
- Display metrics in progress bar
- Display the parameter count by layer
- Force training for min or max epochs
- Inspect gradient norms
- Learning rate annealing
- Make model overfit on subset of data
- Multiple optimizers (like GANs)
- Set how much of the training set to check (1-100%)
- Show progress bar
- training_step function

**Validation loop**    

- Display metrics in progress bar
- Set how much of the validation set to check (1-100%)
- Set validation check frequency within 1 training epoch (1-100%)
- validation_step function
- Why does validation run first for 5 steps?

**Distributed training**    

- Single-gpu      
- Multi-gpu      
- Multi-node   
- 16-bit mixed precision

**Checkpointing**    

- Model saving
- Model loading 

**Computing cluster (SLURM)**    

- Automatic checkpointing   
- Automatic saving, loading  
- Running grid search on a cluster 
- Walltime auto-resubmit   