import os
import shutil

from tests_cloud import STORAGE_DIR


# TODO: make this as a fixture
def cleanup():
    if os.getenv("LIGHTNING_MODEL_STORE_TESTING") and os.path.isdir(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
