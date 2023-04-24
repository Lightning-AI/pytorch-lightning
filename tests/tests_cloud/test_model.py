import os

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.store import download_model, load_model, upload_model
from lightning.store.save import __STORAGE_DIR_NAME
from tests_cloud import _API_KEY, _PROJECT_ID, _USERNAME


@pytest.mark.parametrize("pbar", [True, False])
def test_model(lit_home, pbar, model_name: str = "boring_model", version: str = "latest"):
    upload_model(model_name, model=BoringModel(), api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}", progress_bar=pbar)
    assert os.path.isdir(os.path.join(lit_home, __STORAGE_DIR_NAME, _USERNAME, model_name, version))

    model = load_model(f"{_USERNAME}/{model_name}")
    assert model is not None


def test_only_weights(lit_home, model_name: str = "boring_model_only_weights", version: str = "latest"):
    model = BoringModel()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
    upload_model(model_name, model=model, weights_only=True, api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}")
    assert os.path.isdir(os.path.join(lit_home, __STORAGE_DIR_NAME, _USERNAME, model_name, version))

    model_with_weights = load_model(f"{_USERNAME}/{model_name}", load_weights=True, model=model)
    assert model_with_weights is not None
    assert model_with_weights.state_dict() is not None


def test_checkpoint_path(lit_home, model_name: str = "boring_model_only_checkpoint_path", version: str = "latest"):
    model = BoringModel()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint("tmp.ckpt")
    upload_model(model_name, checkpoint_path="tmp.ckpt", api_key=_API_KEY, project_id=_PROJECT_ID)

    download_model(f"{_USERNAME}/{model_name}")
    assert os.path.isdir(os.path.join(lit_home, __STORAGE_DIR_NAME, _USERNAME, model_name, version))

    ckpt = load_model(f"{_USERNAME}/{model_name}", load_checkpoint=True, model=model)
    assert ckpt is not None
