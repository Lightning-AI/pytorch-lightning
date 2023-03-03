import inspect
import os
import tempfile

from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.store import download_model, upload_model
from lightning.store.save import __STORAGE_DIR_NAME
from tests_cloud import _API_KEY, _PROJECT_ID, _PROJECT_ROOT, _TEST_ROOT, _USERNAME


def test_source_code_implicit(lit_home, model_name: str = "model_test_source_code_implicit"):
    upload_model(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}")
    assert os.path.isfile(
        os.path.join(
            lit_home,
            __STORAGE_DIR_NAME,
            _USERNAME,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_saving_disabled(lit_home, model_name: str = "model_test_source_code_dont_save"):
    upload_model(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID, save_code=False)

    download_model(f"{_USERNAME}/{model_name}")
    assert not os.path.isfile(
        os.path.join(
            lit_home,
            __STORAGE_DIR_NAME,
            _USERNAME,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_explicit_relative_folder(lit_home, model_name: str = "model_test_source_code_explicit_relative"):
    upload_model(model_name, model=BoringModel(), source_code_path=_TEST_ROOT, api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            lit_home,
            __STORAGE_DIR_NAME,
            _USERNAME,
            model_name,
            "latest",
            os.path.basename(os.path.abspath(_TEST_ROOT)),
        )
    )


def test_source_code_explicit_absolute_folder(
    lit_home, model_name: str = "model_test_source_code_explicit_absolute_path"
):
    # TODO: unify with above `test_source_code_explicit_relative_folder`
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_upload_path = os.path.abspath(tmpdir)
        upload_model(
            model_name, model=BoringModel(), source_code_path=dir_upload_path, api_key=_API_KEY, project_id=_PROJECT_ID
        )

    download_model(f"{_USERNAME}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            lit_home,
            __STORAGE_DIR_NAME,
            _USERNAME,
            model_name,
            "latest",
            os.path.basename(os.path.abspath(dir_upload_path)),
        )
    )


def test_source_code_explicit_file(lit_home, model_name: str = "model_test_source_code_explicit_file"):
    file_name = os.path.join(_PROJECT_ROOT, "setup.py")
    upload_model(model_name, model=BoringModel(), source_code_path=file_name, api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}")

    assert os.path.isfile(
        os.path.join(
            lit_home,
            __STORAGE_DIR_NAME,
            _USERNAME,
            model_name,
            "latest",
            os.path.basename(file_name),
        )
    )
