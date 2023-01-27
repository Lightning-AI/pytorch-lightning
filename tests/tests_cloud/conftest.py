import importlib
import sys
import tempfile

import pytest

import lightning
import lightning.store.save


@pytest.fixture(scope="function", autouse=True)
def clean_home(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdirname:
        monkeypatch.setattr(lightning.store.save, "_LIGHTNING_STORAGE_DIR", tmpdirname)
        importlib.reload(sys.modules["lightning.store.save"])
        yield tmpdirname
