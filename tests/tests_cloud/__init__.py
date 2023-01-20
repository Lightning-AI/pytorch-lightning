import os

_LIGHTNING_DIR = f"{os.path.expanduser('~')}/.lightning"

if os.getenv("LIGHTNING_MODEL_STORE_TESTING") == "1":
    STORAGE_DIR = f"{_LIGHTNING_DIR}/lightning_test_model_store/"
else:
    from lightning.store.utils import _LIGHTNING_STORAGE_DIR as STORAGE_DIR

_USERNAME = os.getenv("API_USERNAME", "")
if not _USERNAME:
    raise ValueError(
        "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
    )

__all__ = ["STORAGE_DIR"]
