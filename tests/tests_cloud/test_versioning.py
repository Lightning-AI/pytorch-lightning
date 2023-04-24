import os
import platform

import pytest

from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.store.cloud_api import download_model, upload_model
from lightning.store.save import __STORAGE_DIR_NAME
from tests_cloud import _API_KEY, _PROJECT_ID, _USERNAME


def assert_download_successful(lit_home, username, model_name, version):
    folder_name = os.path.join(lit_home, __STORAGE_DIR_NAME, username, model_name, version)
    assert os.path.isdir(folder_name), f"Folder name: {folder_name} doesn't exist."
    assert len(os.listdir(folder_name)) != 0


@pytest.mark.parametrize(
    ("case", "expected_case"),
    (
        [
            ("1.0.0", "version_1_0_0"),
            ("0.0.1", "version_0_0_1"),
            ("latest", "version_latest"),
            ("1.0", "version_1_0"),
            ("1", "version_1"),
            ("0.1", "version_0_1"),
            ("", "version_latest"),
        ]
    ),
)
def test_versioning_valid_case(lit_home, case, expected_case, model_name: str = "boring_model_versioning"):
    upload_model(model_name, version=case, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)
    download_model(f"{_USERNAME}/{model_name}", version=case)
    assert_download_successful(lit_home, _USERNAME, model_name, expected_case)


@pytest.mark.parametrize(
    "case",
    (
        [
            " version with spaces ",
            "*",
            # "#", <-- TODO: Add it back later
            "¡",
            "©",
        ]
    ),
)
def test_versioning_invalid_case(lit_home, case, model_name: str = "boring_model_versioning"):
    with pytest.raises(ConnectionRefusedError):
        upload_model(model_name, version=case, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    error = OSError if case == "*" and platform.system() == "Windows" else ConnectionRefusedError
    with pytest.raises(error):
        download_model(f"{_USERNAME}/{model_name}", version=case)
