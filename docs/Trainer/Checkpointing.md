Lightning can automate saving and loading checkpoints.

---
### Model saving
To enable checkpointing, define the checkpoint callback and give it to the trainer.

``` {.python}
from pytorch_lightning.utils.pt_callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='/path/to/store/weights.ckpt',
    save_best_only=True,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = Trainer(checkpoint_callback=checkpoint_callback)
```

---
### Restoring training session 
You might want to not only load a model but also continue training it. Use this method to
restore the trainer state as well. This will continue from the epoch and global step you last left off.  
However, the dataloaders will start from the first batch again (if you shuffled it shouldn't matter).   

Lightning will restore the session if you pass an experiment with the same version and there's a saved checkpoint.   
``` {.python}
from test_tube import Experiment

exp = Experiment(version=a_previous_version_with_a_saved_checkpoint)
Trainer(experiment=exp)

trainer = Trainer(checkpoint_callback=checkpoint_callback)
# the trainer is now restored
```



