import os
import shutil

from lightning.store.save import _LIGHTNING_STORAGE_DIR


# TODO: make this as a fixture
def cleanup():
    # todo: `LIGHTNING_MODEL_STORE_TESTING` is nor working as intended,
    #  so the fixture shall create temp folder and map it home...
    if os.getenv("LIGHTNING_MODEL_STORE_TESTING") and os.path.isdir(_LIGHTNING_STORAGE_DIR):
        shutil.rmtree(_LIGHTNING_STORAGE_DIR)
