import inspect
import os
import tempfile

from tests_app.model2cloud.utils import cleanup

from lightning.app.model2cloud import download_from_lightning_cloud, to_lightning_cloud
from pytorch_lightning.demos.boring_classes import BoringModel

if os.getenv("LIGHTNING_MODEL_STORE_TESTING"):
    from tests_app.model2cloud.constants import LIGHTNING_TEST_STORAGE_DIR as LIGHTNING_STORAGE_DIR
else:
    from lightning.app.model2cloud import LIGHTNING_STORAGE_DIR


def test_source_code_implicit():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    username = os.getenv("API_USERNAME")
    model_name = "model_test_source_code_implicit"
    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")
    assert os.path.isfile(
        os.path.join(
            LIGHTNING_STORAGE_DIR,
            username,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_saving_disabled():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "model_test_source_code_dont_save"
    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
        save_code=False,
    )

    download_from_lightning_cloud(f"{username}/{model_name}")
    assert not os.path.isfile(
        os.path.join(
            LIGHTNING_STORAGE_DIR,
            username,
            model_name,
            "latest",
            str(os.path.basename(inspect.getsourcefile(BoringModel))),
        )
    )


def test_source_code_explicit_relative_folder():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    username = os.getenv("API_USERNAME")
    model_name = "model_test_source_code_explicit_relative"
    dir_upload_path = f"../{os.path.basename(os.getcwd())}/tests/"
    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        source_code_path=dir_upload_path,
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            LIGHTNING_STORAGE_DIR,
            username,
            model_name,
            "latest",
            os.path.basename(os.path.abspath(dir_upload_path)),
        )
    )


def test_source_code_explicit_absolute_folder():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    username = os.getenv("API_USERNAME")
    model_name = "model_test_source_code_explicit_absolute_path"
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_upload_path = os.path.abspath(tmpdir)
        to_lightning_cloud(
            model_name,
            model=BoringModel(),
            source_code_path=dir_upload_path,
            api_key=os.getenv("API_KEY", ""),
            project_id=os.getenv("PROJECT_ID", ""),
        )

    download_from_lightning_cloud(f"{username}/{model_name}")

    assert os.path.isdir(
        os.path.join(
            LIGHTNING_STORAGE_DIR,
            username,
            model_name,
            "latest",
            os.path.basename(os.path.abspath(dir_upload_path)),
        )
    )


def test_source_code_explicit_file():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    username = os.getenv("API_USERNAME")
    model_name = "model_test_source_code_explicit_file"
    file_name = os.path.abspath("setup.py")
    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        source_code_path=file_name,
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")

    assert os.path.isfile(
        os.path.join(
            LIGHTNING_STORAGE_DIR,
            username,
            model_name,
            "latest",
            os.path.basename(file_name),
        )
    )
