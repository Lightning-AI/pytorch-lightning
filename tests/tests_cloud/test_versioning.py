import os
import platform

import pytest
from tests_cloud import STORAGE_DIR
from tests_cloud.utils import cleanup

from lightning.store.cloud_api import download_from_lightning_cloud, to_lightning_cloud
from pytorch_lightning.demos.boring_classes import BoringModel


def assert_download_successful(username, model_name, version):
    folder_name = os.path.join(
        STORAGE_DIR,
        username,
        model_name,
        version,
    )
    assert os.path.isdir(folder_name), f"Folder name: {folder_name} doesn't exist."
    assert len(os.listdir(folder_name)) != 0


@pytest.mark.parametrize(
    (
        "case",
        "expected_case",
    ),
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
def test_versioning_valid_case(case, expected_case):
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model_versioning"

    api_key = os.getenv("API_KEY", "")
    to_lightning_cloud(
        model_name,
        version=case,
        model=BoringModel(),
        api_key=api_key,
        project_id=os.getenv("PROJECT_ID", ""),
    )
    download_from_lightning_cloud(f"{username}/{model_name}", version=case)
    assert_download_successful(username, model_name, expected_case)


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
def test_versioning_invalid_case(case):
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model_versioning"

    with pytest.raises(AssertionError):
        api_key = os.getenv("API_KEY", "")
        to_lightning_cloud(
            model_name,
            version=case,
            model=BoringModel(),
            api_key=api_key,
            project_id=os.getenv("PROJECT_ID", ""),
        )

    error = OSError if case == "*" and platform.system() == "Windows" else AssertionError
    with pytest.raises(error):
        download_from_lightning_cloud(f"{username}/{model_name}", version=case)
