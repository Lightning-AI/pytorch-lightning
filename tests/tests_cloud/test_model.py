import os

from tests_cloud import _API_KEY, _PROJECT_ID, _USERNAME, STORAGE_DIR
from tests_cloud.helpers import cleanup

import pytorch_lightning as pl
from lightning.store import download_from_lightning_cloud, load_from_lightning_cloud, to_lightning_cloud
from pytorch_lightning.demos.boring_classes import BoringModel


def test_model(model_name: str = "boring_model", version: str = "latest"):
    cleanup()

    to_lightning_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert os.path.isdir(os.path.join(STORAGE_DIR, _USERNAME, model_name, version))

    model = load_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert model is not None


def test_model_without_progress_bar(model_name: str = "boring_model", version: str = "latest"):
    cleanup()

    to_lightning_cloud(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID, progress_bar=False)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}", progress_bar=False)
    assert os.path.isdir(os.path.join(STORAGE_DIR, _USERNAME, model_name, version))

    model = load_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert model is not None


def test_only_weights(model_name: str = "boring_model_only_weights", version: str = "latest"):
    cleanup()

    model = BoringModel()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    to_lightning_cloud(model_name, model=model, weights_only=True, api_key=_API_KEY, project_id=_PROJECT_ID)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert os.path.isdir(os.path.join(STORAGE_DIR, _USERNAME, model_name, version))

    model_with_weights = load_from_lightning_cloud(f"{_USERNAME}/{model_name}", load_weights=True, model=model)
    assert model_with_weights is not None
    assert model_with_weights.state_dict() is not None


def test_checkpoint_path(model_name: str = "boring_model_only_checkpoint_path", version: str = "latest"):
    cleanup()

    model = BoringModel()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint("tmp.ckpt")
    to_lightning_cloud(model_name, checkpoint_path="tmp.ckpt", api_key=_API_KEY, project_id=_PROJECT_ID)

    download_from_lightning_cloud(f"{_USERNAME}/{model_name}")
    assert os.path.isdir(os.path.join(STORAGE_DIR, _USERNAME, model_name, version))

    ckpt = load_from_lightning_cloud(f"{_USERNAME}/{model_name}", load_checkpoint=True, model=model)
    assert ckpt is not None
