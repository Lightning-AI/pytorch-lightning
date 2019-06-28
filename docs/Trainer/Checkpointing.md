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



