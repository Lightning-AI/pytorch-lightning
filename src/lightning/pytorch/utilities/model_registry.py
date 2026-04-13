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
import os
import re
from typing import Optional

from lightning_utilities import module_available

import lightning.pytorch as pl
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.types import _PATH

# skip these test on Windows as the path notation differ
if _IS_WINDOWS:
    __doctest_skip__ = ["_determine_model_folder"]


def _is_registry(text: Optional[_PATH]) -> bool:
    """Check if a string equals 'registry' or starts with 'registry:'.

    Args:
        text: The string to check

    >>> _is_registry("registry")
    True
    >>> _is_registry("REGISTRY:model-name")
    True
    >>> _is_registry("something_registry")
    False
    >>> _is_registry("")
    False

    """
    if not isinstance(text, str):
        return False

    # Pattern matches exactly 'registry' or 'registry:' followed by any characters
    pattern = r"^registry(:.*|$)"
    return bool(re.match(pattern, text.lower()))


def _parse_registry_model_version(ckpt_path: Optional[_PATH]) -> tuple[str, str]:
    """Parse the model version from a registry path.

    Args:
        ckpt_path: The checkpoint path

    Returns:
        string name and version of the model

    >>> _parse_registry_model_version("registry:model-name:version:1.0")
    ('model-name', '1.0')
    >>> _parse_registry_model_version("registry:model-name")
    ('model-name', '')
    >>> _parse_registry_model_version("registry:VERSION:v2")
    ('', 'v2')

    """
    if not ckpt_path or not _is_registry(ckpt_path):
        raise ValueError(f"Invalid registry path: {ckpt_path}")

    # Split the path by ':'
    parts = str(ckpt_path).split(":")
    # Default values
    model_name, version = "", ""

    # Extract the model name and version based on the parts
    if len(parts) >= 2 and parts[1].lower() != "version":
        model_name = parts[1]
    if len(parts) == 3 and parts[1].lower() == "version":
        version = parts[2]
    elif len(parts) == 4 and parts[2].lower() == "version":
        version = parts[3]

    return model_name, version


def _determine_model_name(ckpt_path: Optional[_PATH], default_model_registry: Optional[str]) -> str:
    """Determine the model name from the checkpoint path.

    Args:
        ckpt_path: The checkpoint path
        default_model_registry: The default model registry

    Returns:
        string name of the model with optional version

    >>> _determine_model_name("registry:model-name:version:1.0", "default-model")
    'model-name:1.0'
    >>> _determine_model_name("registry:model-name", "default-model")
    'model-name'
    >>> _determine_model_name("registry:version:v2", "default-model")
    'default-model:v2'

    """
    # try to find model and version
    model_name, model_version = _parse_registry_model_version(ckpt_path)
    # omitted model name try to use the model registry from Trainer
    if not model_name and default_model_registry:
        model_name = default_model_registry
    if not model_name:
        raise ValueError(f"Invalid model registry: '{ckpt_path}'")
    model_registry = model_name
    model_registry += f":{model_version}" if model_version else ""
    return model_registry


def _determine_model_folder(model_name: str, default_root_dir: str) -> str:
    """Determine the local model folder based on the model registry.

    Args:
        model_name: The model name
        default_root_dir: The default root directory

    Returns:
        string path to the local model folder

    >>> _determine_model_folder("model-name", "/path/to/root")
    '/path/to/root/model-name'
    >>> _determine_model_folder("model-name:1.0", "/path/to/root")
    '/path/to/root/model-name_1.0'

    """
    if not model_name:
        raise ValueError(f"Invalid model registry: '{model_name}'")
    # download the latest checkpoint from the model registry
    model_name = model_name.replace("/", "_")
    model_name = model_name.replace(":", "_")
    local_model_dir = os.path.join(default_root_dir, model_name)
    return local_model_dir


def find_model_local_ckpt_path(
    ckpt_path: Optional[_PATH], default_model_registry: Optional[str], default_root_dir: str
) -> str:
    """Find the local checkpoint path for a model."""
    model_registry = _determine_model_name(ckpt_path, default_model_registry)
    local_model_dir = _determine_model_folder(model_registry, default_root_dir)

    # todo: resolve if there are multiple checkpoints
    folder_files = [fn for fn in os.listdir(local_model_dir) if fn.endswith(".ckpt")]
    if not folder_files:
        raise RuntimeError(f"Parsing files from downloaded model: {model_registry}")
    # print(f"local RANK {self.trainer.local_rank}: using model files: {folder_files}")
    return os.path.join(local_model_dir, folder_files[0])


def download_model_from_registry(ckpt_path: Optional[_PATH], trainer: "pl.Trainer") -> None:
    """Download a model from the Lightning Model Registry."""
    if trainer.local_rank == 0:
        if not module_available("litmodels"):
            raise ImportError(
                "The `litmodels` package is not installed. Please install it with `pip install litmodels`."
            )

        from litmodels import download_model

        model_registry = _determine_model_name(ckpt_path, trainer._model_registry)
        local_model_dir = _determine_model_folder(model_registry, trainer.default_root_dir)

        # print(f"Rank {self.trainer.local_rank} downloads model checkpoint '{model_registry}'")
        model_files = download_model(model_registry, download_dir=local_model_dir)
        # print(f"Model checkpoint '{model_registry}' was downloaded to '{local_model_dir}'")
        if not model_files:
            raise RuntimeError(f"Download model failed - {model_registry}")

    trainer.strategy.barrier("download_model_from_registry")
