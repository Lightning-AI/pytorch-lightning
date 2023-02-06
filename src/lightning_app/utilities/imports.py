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
"""General utilities."""

import functools
import os
import warnings
from typing import List, Union

from lightning_utilities.core.imports import module_available
from packaging.requirements import Marker, Requirement

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata  # type: ignore


def _get_extras(extras: str) -> str:
    """Get the given extras as a space delimited string.

    Used by the platform to install cloud extras in the cloud.
    """
    from lightning_app import __package_name__

    requirements = {r: Requirement(r) for r in metadata.requires(__package_name__)}
    marker = Marker(f'extra == "{extras}"')
    requirements = [r for r, req in requirements.items() if str(req.marker) == str(marker)]

    if requirements:
        requirements = [f"'{r.split(';')[0].strip()}'" for r in requirements]
        return " ".join(requirements)
    return ""


def requires(module_paths: Union[str, List]):
    if not isinstance(module_paths, list):
        module_paths = [module_paths]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            unavailable_modules = [f"'{module}'" for module in module_paths if not module_available(module)]
            if any(unavailable_modules):
                is_lit_testing = bool(int(os.getenv("LIGHTING_TESTING", "0")))
                msg = f"Required dependencies not available. Please run: pip install {' '.join(unavailable_modules)}"
                if is_lit_testing:
                    warnings.warn(msg)
                else:
                    raise ModuleNotFoundError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# TODO: Automatically detect dependencies
def _is_redis_available() -> bool:
    return module_available("redis")


def _is_torch_available() -> bool:
    return module_available("torch")


def _is_pytorch_lightning_available() -> bool:
    return module_available("pytorch_lightning")


def _is_torchvision_available() -> bool:
    return module_available("torchvision")


def _is_json_argparse_available() -> bool:
    return module_available("jsonargparse")


def _is_streamlit_available() -> bool:
    return module_available("streamlit")


def _is_param_available() -> bool:
    return module_available("param")


def _is_streamlit_tensorboard_available() -> bool:
    return module_available("streamlit_tensorboard")


def _is_starsessions_available() -> bool:
    return module_available("starsessions")


def _is_gradio_available() -> bool:
    return module_available("gradio")


def _is_lightning_flash_available() -> bool:
    return module_available("flash")


def _is_pil_available() -> bool:
    return module_available("PIL")


def _is_numpy_available() -> bool:
    return module_available("numpy")


def _is_docker_available() -> bool:
    return module_available("docker")


def _is_jinja2_available() -> bool:
    return module_available("jinja2")


def _is_playwright_available() -> bool:
    return module_available("playwright")


def _is_s3fs_available() -> bool:
    return module_available("s3fs")


def _is_sqlmodel_available() -> bool:
    return module_available("sqlmodel")


def _is_aiohttp_available() -> bool:
    return module_available("aiohttp")


_CLOUD_TEST_RUN = bool(os.getenv("CLOUD", False))
