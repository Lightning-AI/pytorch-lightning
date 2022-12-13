import os
import shutil

from tests_app.model2cloud.constants import LIGHTNING_TEST_STORAGE_DIR


def cleanup():
    if os.getenv("LIGHTNING_MODEL_STORE_TESTING"):
        if os.path.isdir(LIGHTNING_TEST_STORAGE_DIR):
            shutil.rmtree(LIGHTNING_TEST_STORAGE_DIR)
