import os

from tests_app.model2cloud.utils import cleanup

from lightning.app.model2cloud import download_from_lightning_cloud, to_lightning_cloud
from pytorch_lightning.demos.boring_classes import BoringModel

if os.getenv("LIGHTNING_MODEL_STORE_TESTING"):
    from tests_app.model2cloud.constants import LIGHTNING_TEST_STORAGE_DIR as LIGHTNING_STORAGE_DIR
else:
    from lightning.app.model2cloud.utils import LIGHTNING_STORAGE_DIR


def test_requirements_as_a_file():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    requirements_file_path = "tests/requirements.txt"
    version = "latest"
    model_name = "boring_model"

    to_lightning_cloud(
        model_name,
        version=version,
        model=BoringModel(),
        requirements_file_path=requirements_file_path,
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")

    req_folder_path = os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version)
    assert os.path.isdir(req_folder_path)
    assert "requirements.txt" in os.listdir(req_folder_path)


def test_requirements_as_a_list():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    version = "1.0.0"
    model_name = "boring_model"
    requirements_list = ["pytorch_lightning==1.7.7", "lightning"]

    to_lightning_cloud(
        model_name,
        version=version,
        model=BoringModel(),
        requirements=requirements_list,
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}", version=version)

    req_folder_path = os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version)
    assert os.path.isdir(req_folder_path)
    assert "requirements.txt" in os.listdir(req_folder_path)

    with open(f"{req_folder_path}/requirements.txt") as req_file:
        reqs = req_file.readlines()
        reqs = [req.strip("\n") for req in reqs]

    assert requirements_list == reqs
