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
"""Warning-related utilities."""
import importlib.util
import os
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Type, Union

from lightning.fabric.utilities.rank_zero import LightningDeprecationWarning

# enable our warnings
warnings.simplefilter("default", category=LightningDeprecationWarning)


def _wrap_formatwarning(default_format_warning: Callable) -> Callable:
    """Custom formatting that avoids an extra line in case warnings are emitted from the `rank_zero`-functions."""

    @wraps(default_format_warning)
    def wrapper(
        message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str] = None
    ) -> str:
        if _is_path_in_lightning(Path(filename)):
            # The warning originates from the Lightning package
            return f"{filename}:{lineno}: {message}\n"
        return default_format_warning(message, category, filename, lineno, line)

    return wrapper


warnings.formatwarning = _wrap_formatwarning(warnings.formatwarning)


class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""


def _is_path_in_lightning(path: Path) -> bool:
    """Checks whether the given path is a subpath of the Lightning package."""
    path = Path(path).absolute()
    lightning_roots = _get_lightning_package_roots()
    for lightning_root in lightning_roots:
        if path.drive != lightning_root.drive:  # handle windows
            continue
        common_path = Path(os.path.commonpath([path, lightning_root]))
        if common_path.name in ("lightning", "lightning_fabric", "pytorch_lightning"):
            return True
    return False


def _get_lightning_package_roots() -> List[Path]:
    """Returns the absolute path to each of the Lightning packages."""
    roots = []
    for name in ("lightning", "lightning_fabric", "pytorch_lightning"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin is not None:
            roots.append(Path(spec.origin).parent.absolute())
    return roots
