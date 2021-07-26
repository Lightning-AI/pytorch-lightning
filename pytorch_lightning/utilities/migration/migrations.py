import pytorch_lightning as pl
from pytorch_lightning.utilities.migration.base import set_version, should_upgrade


# v0.10.0
def migrate_model_checkpoint_early_stopping(checkpoint: dict) -> dict:
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    keys_mapping = {
        "checkpoint_callback_best_model_score": (ModelCheckpoint, "best_model_score"),
        "checkpoint_callback_best_model_path": (ModelCheckpoint, "best_model_path"),
        "checkpoint_callback_best": (ModelCheckpoint, "best_model_score"),
        "early_stop_callback_wait": (EarlyStopping, "wait_count"),
        "early_stop_callback_patience": (EarlyStopping, "patience"),
    }
    checkpoint["callbacks"] = checkpoint.get("callbacks") or {}

    for key, new_path in keys_mapping.items():
        if key in checkpoint:
            value = checkpoint[key]
            callback_type, callback_key = new_path
            checkpoint["callbacks"][callback_type] = checkpoint["callbacks"].get(callback_type) or {}
            checkpoint["callbacks"][callback_type][callback_key] = value
            del checkpoint[key]
    return checkpoint


def migrate_checkpoint(checkpoint: dict):
    """Applies all the above migrations in order."""
    if should_upgrade(checkpoint, "0.10.0"):
        migrate_model_checkpoint_early_stopping(checkpoint)
    set_version(checkpoint, pl.__version__)
    return checkpoint
