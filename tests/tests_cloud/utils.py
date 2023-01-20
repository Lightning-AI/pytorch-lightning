import os
import shutil

from tests_cloud import LIGHTNING_TEST_STORAGE_DIR


def cleanup():
    if os.getenv("LIGHTNING_MODEL_STORE_TESTING") and os.path.isdir(LIGHTNING_TEST_STORAGE_DIR):
        shutil.rmtree(LIGHTNING_TEST_STORAGE_DIR)
