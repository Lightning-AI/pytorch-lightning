Lightning can automate saving and loading checkpoints.

---
### Model saving
To enable checkpointing, define the checkpoint callback

``` {.python}
from pytorch_lightning.utils.pt_callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='/path/to/store/weights.ckpt',
    save_function=None,
    save_best_only=not hparams.keep_all_checkpoints,
    verbose=True,
    monitor=hparams.model_save_monitor_value,
    mode=hparams.model_save_monitor_mode
)
```