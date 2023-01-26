import inspect
import os
import tempfile

from tests_cloud import _API_KEY, _PROJECT_ID, _PROJECT_ROOT, _TEST_ROOT, _USERNAME
from tests_cloud.helpers import cleanup

from lightning.store import download_from_cloud, upload_to_cloud
from lightning.store.save import _LIGHTNING_STORAGE_DIR
from pytorch_lightning.demos.boring_classes import BoringModel


def test_source_code_implicit(model_name: str = "model_test_source_code_implicit"):
    cleanup()

    upload_to_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    download_from_cloud(f"{_USERNAME}/{model_name}")
    assert os.path.isfile(
        os.path.join(
            _LIGHTNING_STORAGE_DIR,
            _USERNAME,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_saving_disabled(model_name: str = "model_test_source_code_dont_save"):
    cleanup()

    upload_to_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID, save_code=False)

    download_from_cloud(f"{_USERNAME}/{model_name}")
    assert not os.path.isfile(
        os.path.join(
            _LIGHTNING_STORAGE_DIR,
            _USERNAME,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_explicit_relative_folder(model_name: str = "model_test_source_code_explicit_relative"):
    cleanup()

    dir_upload_path = _TEST_ROOT
    upload_to_cloud(
        model_name, model=BoringModel(), source_code_path=dir_upload_path, api_key=_API_KEY, project_id=_PROJECT_ID
    )

    download_from_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            _LIGHTNING_STORAGE_DIR, _USERNAME, model_name, "latest", os.path.basename(os.path.abspath(dir_upload_path))
        )
    )


def test_source_code_explicit_absolute_folder(model_name: str = "model_test_source_code_explicit_absolute_path"):
    cleanup()

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_upload_path = os.path.abspath(tmpdir)
        upload_to_cloud(
            model_name, model=BoringModel(), source_code_path=dir_upload_path, api_key=_API_KEY, project_id=_PROJECT_ID
        )

    download_from_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            _LIGHTNING_STORAGE_DIR, _USERNAME, model_name, "latest", os.path.basename(os.path.abspath(dir_upload_path))
        )
    )


def test_source_code_explicit_file(model_name: str = "model_test_source_code_explicit_file"):
    cleanup()

    file_name = os.path.join(_PROJECT_ROOT, "setup.py")
    upload_to_cloud(
        model_name, model=BoringModel(), source_code_path=file_name, api_key=_API_KEY, project_id=_PROJECT_ID
    )

    download_from_cloud(f"{_USERNAME}/{model_name}")

    assert os.path.isfile(
        os.path.join(
            _LIGHTNING_STORAGE_DIR,
            _USERNAME,
            model_name,
            "latest",
            os.path.basename(file_name),
        )
    )
