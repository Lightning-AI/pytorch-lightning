import os

from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.store import download_model, upload_model
from lightning.store.save import __STORAGE_DIR_NAME
from tests_cloud import _API_KEY, _PROJECT_ID, _USERNAME


def test_requirements(lit_home, version: str = "1.0.0", model_name: str = "boring_model"):
    requirements_list = ["pytorch_lightning==1.7.7", "lightning"]

    upload_model(
        model_name,
        version=version,
        model=BoringModel(),
        requirements=requirements_list,
        api_key=_API_KEY,
        project_id=_PROJECT_ID,
    )

    download_model(f"{_USERNAME}/{model_name}", version=version)

    req_folder_path = os.path.join(lit_home, __STORAGE_DIR_NAME, _USERNAME, model_name, version)
    assert os.path.isdir(req_folder_path), "missing: %s" % req_folder_path
    assert "requirements.txt" in os.listdir(req_folder_path), "among files: %r" % os.listdir(req_folder_path)

    with open(f"{req_folder_path}/requirements.txt") as req_file:
        reqs = req_file.readlines()
        reqs = [req.strip("\n") for req in reqs]

    assert requirements_list == reqs
