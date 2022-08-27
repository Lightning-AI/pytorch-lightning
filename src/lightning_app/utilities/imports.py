# Copyright The PyTorch Lightning team.
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
import importlib
import os
from importlib.util import find_spec
from typing import List, Union


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@functools.lru_cache()
def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('os.bla')
    False
    >>> _module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError:
        return False
    return True


def requires(module_paths: Union[str, List]):

    if not isinstance(module_paths, list):
        module_paths = [module_paths]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            unavailable_modules = [f"'{module}'" for module in module_paths if not _module_available(module)]
            if any(unavailable_modules) and not bool(int(os.getenv("LIGHTING_TESTING", "0"))):
                raise ModuleNotFoundError(
                    f"Required dependencies not available. Please run: pip install {' '.join(unavailable_modules)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# TODO: Automatically detect dependencies
@functools.lru_cache()
def _is_redis_available() -> bool:
    return _module_available("redis")


@functools.lru_cache()
def _is_torch_available() -> bool:
    return _module_available("torch")


@functools.lru_cache()
def _is_pytorch_lightning_available() -> bool:
    return _module_available("pytorch_lightning")


@functools.lru_cache()
def _is_torchvision_available() -> bool:
    return _module_available("torchvision")


@functools.lru_cache()
def _is_json_argparse_available() -> bool:
    return _module_available("jsonargparse")


@functools.lru_cache()
def _is_streamlit_available() -> bool:
    return _module_available("streamlit")


@functools.lru_cache()
def _is_streamlit_tensorboard_available() -> bool:
    return _module_available("streamlit_tensorboard")


@functools.lru_cache()
def _is_starsessions_available() -> bool:
    return _module_available("starsessions")


@functools.lru_cache()
def _is_gradio_available() -> bool:
    return _module_available("gradio")


@functools.lru_cache()
def _is_lightning_flash_available() -> bool:
    return _module_available("flash")


@functools.lru_cache()
def _is_pil_available() -> bool:
    return _module_available("PIL")


@functools.lru_cache()
def _is_numpy_available() -> bool:
    return _module_available("numpy")


@functools.lru_cache()
def _is_docker_available() -> bool:
    return _module_available("docker")


@functools.lru_cache()
def _is_jinja2_available() -> bool:
    return _module_available("jinja2")


@functools.lru_cache()
def _is_playwright_available() -> bool:
    return _module_available("playwright")


@functools.lru_cache()
def _is_s3fs_available() -> bool:
    return _module_available("s3fs")


_CLOUD_TEST_RUN = bool(os.getenv("CLOUD", False))
