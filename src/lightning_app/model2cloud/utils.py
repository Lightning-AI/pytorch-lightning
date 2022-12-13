import json
import os
from enum import Enum


class stage(Enum):
    UPLOAD = 0
    LOAD = 1
    DOWNLOAD = 2


LIGHTNING_DIR = f"{os.path.expanduser('~')}/.lightning"
LIGHTNING_STORAGE_FILE = f"{LIGHTNING_DIR}/.lightning_model_storage"
LIGHTNING_STORAGE_DIR = f"{LIGHTNING_DIR}/lightning_model_store"
LIGHTNING_CLOUD_URL = os.getenv("LIGHTNING_CLOUD_URL", default="https://lightning.ai")


def _check_version(version: str):
    allowed_chars = "0123456789."
    if version == "latest":
        return True
    for version_char in version:
        if version_char not in allowed_chars:
            return False
    return True


def _split_name(name: str, version: str, l_stage: stage):
    if l_stage == stage.UPLOAD:
        username = ""
        model_name = name
    else:
        username, model_name = name.split("/")

    return username, model_name, version


def split_name(name: str, version: str, l_stage: stage):
    username, model_name, version = _split_name(name, version, l_stage)

    return (
        username,
        model_name,
        version,
    )


def get_model_data(name: str, version: str):
    username, model_name, version = split_name(name, version, stage.LOAD)

    assert os.path.exists(
        LIGHTNING_STORAGE_FILE
    ), f"ERROR: Could not find {LIGHTNING_STORAGE_FILE} (to be generated after download_from_lightning_cloud(...))"

    with open(LIGHTNING_STORAGE_FILE) as storage_file:
        storage_data = json.load(storage_file)

    assert username in storage_data, (
        f"No data found for the given username {username}. Make sure to call"
        " `download_from_lightning_cloud` before loading"
    )
    user_data = storage_data[username]

    assert model_name in user_data, (
        f"No data found for the given model name: {model_name} for the given"
        f" username: {username}. Make sure to call `download_from_lightning_cloud` before loading"
    )
    model_data = user_data[model_name]

    version = version or "latest"
    assert (
        version in model_data
    ), f"No data found for the given version: {version}, did you download the model successfully?"
    model_version_data = model_data[version]

    return model_version_data
