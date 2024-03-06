# Copyright The Lightning AI team.

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from unittests import _PATH_ROOT

_PATH_DOCS = os.path.join(_PATH_ROOT, "docs", "source")


@pytest.fixture(scope="session")
def temp_docs():
    """Create a dummy documentation folder."""
    # create a folder for docs
    docs_folder = Path(tempfile.mkdtemp())
    # copy all real docs from _PATH_DOCS to local temp_docs
    for root, _, files in os.walk(_PATH_DOCS):
        for file in files:
            fpath = os.path.join(root, file)
            temp_path = docs_folder / os.path.relpath(fpath, _PATH_DOCS)
            temp_path.parent.mkdir(exist_ok=True, parents=True)
            with open(fpath, "rb") as fopen:
                temp_path.write_bytes(fopen.read())
    yield str(docs_folder)
    # remove the folder
    shutil.rmtree(docs_folder.parent, ignore_errors=True)
