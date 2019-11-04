Lightning can automate saving and loading checkpoints.

---

### Model saving
Checkpointing is enabled by default to the current working directory.
To change the checkpoint path pass in :
```python
Trainer(default_save_path='/your/path/to/save/checkpoints')
```

To modify the behavior of checkpointing pass in your own callback.

```{.python}
from pytorch_lightning.callbacks import ModelCheckpoint

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=-1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

trainer = Trainer(checkpoint_callback=checkpoint_callback)
```

---

### Restoring training session

You might want to not only load a model but also continue training it. Use this method to
restore the trainer state as well. This will continue from the epoch and global step you last left off.  
However, the dataloaders will start from the first batch again (if you shuffled it shouldn't matter).

Lightning will restore the session if you pass a logger with the same version and there's a saved checkpoint.   
``` {.python}
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

logger = TestTubeLogger(
    save_dir='./savepath',
    version=1  # An existing version with a saved checkpoint
)
trainer = Trainer(
    logger=logger,
    default_save_path='./savepath'
)

# this fit call loads model weights and trainer state
# the trainer continues seamlessly from where you left off
# without having to do anything else.
trainer.fit(model)
```

The trainer restores:

- global_step
- current_epoch
- All optimizers
- All lr_schedulers
- Model weights

You can even change the logic of your model as long as the weights and "architecture" of
the system isn't different. If you add a layer, for instance, it might not work.

At a rough level, here's [what happens inside Trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/model_saving.py#L63):

```python

self.global_step = checkpoint['global_step']
self.current_epoch = checkpoint['epoch']

# restore the optimizers
optimizer_states = checkpoint['optimizer_states']
for optimizer, opt_state in zip(self.optimizers, optimizer_states):
    optimizer.load_state_dict(opt_state)

# restore the lr schedulers
lr_schedulers = checkpoint['lr_schedulers']
for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
    scheduler.load_state_dict(lrs_state)

# uses the model you passed into trainer
model.load_state_dict(checkpoint['state_dict'])
```
