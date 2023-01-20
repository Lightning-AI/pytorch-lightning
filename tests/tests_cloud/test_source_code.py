import inspect
import os
import tempfile

from tests_cloud import _API_KEY, _PROJECT_ID, _USERNAME, STORAGE_DIR
from tests_cloud.helpers import cleanup

from lightning.store import download_from_lightning_cloud, to_lightning_cloud
from pytorch_lightning.demos.boring_classes import BoringModel


def test_source_code_implicit(model_name: str = "model_test_source_code_implicit"):
    cleanup()

    to_lightning_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert os.path.isfile(
        os.path.join(
            STORAGE_DIR, _USERNAME, model_name, "latest", str(os.path.basename(inspect.getsourcefile(BoringModel)))
        )
    )


def test_source_code_saving_disabled(model_name: str = "model_test_source_code_dont_save"):
    cleanup()

    to_lightning_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID, save_code=False)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert not os.path.isfile(
        os.path.join(
            STORAGE_DIR, _USERNAME, model_name, "latest", str(os.path.basename(inspect.getsourcefile(BoringModel)))
        )
    )


def test_source_code_explicit_relative_folder(model_name: str = "model_test_source_code_explicit_relative"):
    cleanup()

    dir_upload_path = f"../{os.path.basename(os.getcwd())}/tests/"
    to_lightning_cloud(
        model_name, model=BoringModel(), source_code_path=dir_upload_path, api_key=_API_KEY, project_id=_PROJECT_ID
    )

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(STORAGE_DIR, _USERNAME, model_name, "latest", os.path.basename(os.path.abspath(dir_upload_path)))
    )


def test_source_code_explicit_absolute_folder(model_name: str = "model_test_source_code_explicit_absolute_path"):
    cleanup()

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_upload_path = os.path.abspath(tmpdir)
        to_lightning_cloud(
            model_name, model=BoringModel(), source_code_path=dir_upload_path, api_key=_API_KEY, project_id=_PROJECT_ID
        )

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(STORAGE_DIR, _USERNAME, model_name, "latest", os.path.basename(os.path.abspath(dir_upload_path)))
    )


def test_source_code_explicit_file(model_name: str = "model_test_source_code_explicit_file"):
    cleanup()

    file_name = os.path.abspath("setup.py")
    to_lightning_cloud(
        model_name, model=BoringModel(), source_code_path=file_name, api_key=_API_KEY, project_id=_PROJECT_ID
    )

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isfile(
        os.path.join(
            STORAGE_DIR,
            _USERNAME,
            model_name,
            "latest",
            os.path.basename(file_name),
        )
    )
