import os

from models_cloud import download_from_lightning_cloud, load_from_lightning_cloud, to_lightning_cloud
from tests.utils import cleanup

import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel

if os.getenv("LIGHTNING_MODEL_STORE_TESTING"):
    from tests_app.model2cloud.constants import LIGHTNING_TEST_STORAGE_DIR as LIGHTNING_STORAGE_DIR
else:
    from lightning.app.model2cloud.utils import LIGHTNING_STORAGE_DIR


def test_model():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model"
    version = "latest"

    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")
    assert os.path.isdir(os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version))

    model = load_from_lightning_cloud(f"{username}/{model_name}")
    assert model is not None


def test_model_without_progress_bar():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model"
    version = "latest"

    to_lightning_cloud(
        model_name,
        model=BoringModel(),
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
        progress_bar=False,
    )

    download_from_lightning_cloud(f"{username}/{model_name}", progress_bar=False)
    assert os.path.isdir(os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version))

    model = load_from_lightning_cloud(f"{username}/{model_name}")
    assert model is not None


def test_only_weights():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model_only_weights"
    version = "latest"

    model = BoringModel()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    to_lightning_cloud(
        model_name,
        model=model,
        weights_only=True,
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")
    assert os.path.isdir(os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version))

    model_with_weights = load_from_lightning_cloud(f"{username}/{model_name}", load_weights=True, model=model)
    assert model_with_weights is not None
    assert model_with_weights.state_dict() is not None


def test_checkpoint_path():
    cleanup()
    username = os.getenv("API_USERNAME", "")
    if not username:
        raise ValueError(
            "No API_USERNAME env variable, to test, make sure to add export API_USERNAME='yourusername' before testing"
        )

    model_name = "boring_model_only_checkpoint_path"
    version = "latest"

    model = BoringModel()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint("tmp.ckpt")
    to_lightning_cloud(
        model_name,
        checkpoint_path="tmp.ckpt",
        api_key=os.getenv("API_KEY", ""),
        project_id=os.getenv("PROJECT_ID", ""),
    )

    download_from_lightning_cloud(f"{username}/{model_name}")
    assert os.path.isdir(os.path.join(LIGHTNING_STORAGE_DIR, username, model_name, version))

    ckpt = load_from_lightning_cloud(f"{username}/{model_name}", load_checkpoint=True, model=model)
    assert ckpt is not None
