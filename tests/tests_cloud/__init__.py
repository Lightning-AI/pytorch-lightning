import os

_LIGHTNING_DIR = f"{os.path.expanduser('~')}/.lightning"

if os.getenv("LIGHTNING_MODEL_STORE_TESTING") == "1":
    STORAGE_DIR = f"{_LIGHTNING_DIR}/lightning_test_model_store/"
else:
    from lightning.store.utils import _LIGHTNING_STORAGE_DIR as STORAGE_DIR

__all__ = ["STORAGE_DIR"]
