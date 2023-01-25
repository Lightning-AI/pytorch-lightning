import os

from tests_cloud import _API_KEY, _PROJECT_ID, _PROJECT_ROOT, _USERNAME
from tests_cloud.helpers import cleanup

from lightning.store import download_from_lightning_cloud, to_lightning_cloud
from lightning.store.save import _LIGHTNING_STORAGE_DIR
from pytorch_lightning.demos.boring_classes import BoringModel


def test_requirements_as_a_file(version: str = "latest", model_name: str = "boring_model"):
    cleanup()

    requirements_file_path = os.path.join(_PROJECT_ROOT, "requirements", "app", "base.txt")

    to_lightning_cloud(
        model_name,
        version=version,
        model=BoringModel(),
        requirements_file_path=requirements_file_path,
        api_key=_API_KEY,
        project_id=_PROJECT_ID,
    )

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")

    req_folder_path = os.path.join(_LIGHTNING_STORAGE_DIR, _USERNAME, model_name, version)
    assert os.path.isdir(req_folder_path), "missing: %s" % req_folder_path
    assert "requirements.txt" in os.listdir(req_folder_path), "among files: %r" % os.listdir(req_folder_path)


def test_requirements_as_a_list(version: str = "1.0.0", model_name: str = "boring_model"):
    cleanup()

    requirements_list = ["pytorch_lightning==1.7.7", "lightning"]

    to_lightning_cloud(
        model_name,
        version=version,
        model=BoringModel(),
        requirements=requirements_list,
        api_key=_API_KEY,
        project_id=_PROJECT_ID,
    )

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}", version=version)

    req_folder_path = os.path.join(_LIGHTNING_STORAGE_DIR, _USERNAME, model_name, version)
    assert os.path.isdir(req_folder_path), "missing: %s" % req_folder_path
    assert "requirements.txt" in os.listdir(req_folder_path), "among files: %r" % os.listdir(req_folder_path)

    with open(f"{req_folder_path}/requirements.txt") as req_file:
        reqs = req_file.readlines()
        reqs = [req.strip("\n") for req in reqs]

    assert requirements_list == reqs
