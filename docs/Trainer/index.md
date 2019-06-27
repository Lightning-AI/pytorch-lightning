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

``` {.python}
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment, SlurmCluster

trainer = Trainer(
                 experiment=Experiment,
                 checkpoint_callback=ModelCheckpoint, 
                 early_stop_callback=EarlyStopping,
                 cluster=SlurmCluster,
                 process_position=0,
                 current_gpu_name=0,
                 gpus=None,
                 enable_tqdm=True,
                 overfit_pct=0.0,
                 track_grad_norm=-1,
                 check_val_every_n_epoch=1,
                 fast_dev_run=False,
                 accumulate_grad_batches=1,
                 enable_early_stop=True, max_nb_epochs=5, min_nb_epochs=1,
                 train_percent_check=1.0, 
                 val_percent_check=1.0, 
                 test_percent_check=1.0, 
                 val_check_interval=0.95,
                 log_save_interval=1, add_log_row_interval=1,
                 lr_scheduler_milestones=None,
                 use_amp=False,
                 check_grad_nans=False,
                 amp_level='O2',
                 nb_sanity_val_steps=5):
)
```


Things you can do with the trainer module:

**Training loop**    

- Accumulate gradients
- Check GPU usage
- Check which gradients are nan
- Check validation every n epochs
- Display metrics in progress bar
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