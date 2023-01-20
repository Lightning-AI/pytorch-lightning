# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from enum import Enum


class stage(Enum):
    UPLOAD = 0
    LOAD = 1
    DOWNLOAD = 2


_LIGHTNING_DIR = os.path.join(os.path.expanduser("~"), ".lightning")
_LIGHTNING_STORAGE_FILE = os.path.join(_LIGHTNING_DIR, ".lightning_model_storage")
_LIGHTNING_STORAGE_DIR = os.path.join(_LIGHTNING_DIR, "lightning_model_store")
_LIGHTNING_CLOUD_URL = os.getenv("LIGHTNING_CLOUD_URL", default="https://lightning.ai")


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

    return (username, model_name, version)


def get_model_data(name: str, version: str):
    username, model_name, version = split_name(name, version, stage.LOAD)

    assert os.path.exists(
        _LIGHTNING_STORAGE_FILE
    ), f"ERROR: Could not find {_LIGHTNING_STORAGE_FILE} (to be generated after download_from_lightning_cloud(...))"

    with open(_LIGHTNING_STORAGE_FILE) as storage_file:
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
