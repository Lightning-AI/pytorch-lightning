# Copyright The Lightning AI team.
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
from typing import Tuple

from lightning.store.save import _LIGHTNING_STORAGE_FILE


class stage(Enum):
    UPLOAD = 0
    LOAD = 1
    DOWNLOAD = 2


def _check_version(version: str) -> bool:
    allowed_chars = "0123456789."
    if version == "latest":
        return True
    for version_char in version:
        if version_char not in allowed_chars:
            return False
    return True


def _split_name(name: str, version: str, l_stage: stage) -> Tuple[str, str, str]:
    if l_stage == stage.UPLOAD:
        username = ""
        model_name = name
    else:
        username, model_name = name.split("/")

    return username, model_name, version


def _get_model_data(name: str, version: str):
    username, model_name, version = _split_name(name, version, stage.LOAD)

    if not os.path.exists(_LIGHTNING_STORAGE_FILE):
        raise NotADirectoryError(
            f"Could not find {_LIGHTNING_STORAGE_FILE} (to be generated after download_model(...))"
        )

    with open(_LIGHTNING_STORAGE_FILE) as storage_file:
        storage_data = json.load(storage_file)

    if username not in storage_data:
        raise KeyError(
            f"No data found for the given username {username}. Make sure to call" " `download_model` before loading"
        )
    user_data = storage_data[username]

    if model_name not in user_data:
        raise KeyError(
            f"No data found for the given model name: {model_name} for the given"
            f" username: {username}. Make sure to call `download_model` before loading"
        )
    model_data = user_data[model_name]

    version = version or "latest"
    if version not in model_data:
        raise KeyError(f"No data found for the given version: {version}, did you download the model successfully?")
    model_version_data = model_data[version]

    return model_version_data
