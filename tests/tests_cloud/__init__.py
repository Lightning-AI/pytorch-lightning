import os

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_TEST_ROOT))
_LIGHTNING_DIR = f"{os.path.expanduser('~')}/.lightning"

if os.getenv("LIGHTNING_MODEL_STORE_TESTING") == "1":
    STORAGE_DIR = f"{_LIGHTNING_DIR}/lightning_test_model_store/"
else:
    from lightning.store.utils import _LIGHTNING_STORAGE_DIR as STORAGE_DIR

_USERNAME = os.getenv("API_USERNAME", "")
assert _USERNAME, "No API_USERNAME env variable, make sure to add it before testing"
_API_KEY = os.getenv("API_KEY", "")
assert _API_KEY, "No API_KEY env variable, make sure to add it before testing"
_PROJECT_ID = os.getenv("PROJECT_ID", "")
assert _PROJECT_ID, "No PROJECT_ID env variable, make sure to add it before testing"

__all__ = ["STORAGE_DIR"]
