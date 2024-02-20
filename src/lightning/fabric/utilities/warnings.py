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

import warnings
from pathlib import Path
from typing import Optional, Type, Union

from lightning.fabric.utilities.rank_zero import LightningDeprecationWarning

# enable our warnings
warnings.simplefilter("default", category=LightningDeprecationWarning)
_default_format_warning = warnings.formatwarning


def disable_possible_user_warnings(module: str = "") -> None:
    """Ignore warnings of the category ``PossibleUserWarning`` from Lightning.

    For more granular control over which warnings to ignore, use :func:`warnings.filterwarnings` directly.

    Args:
        module: Name of the module for which the warnings should be ignored (e.g., ``'lightning.pytorch.strategies'``).
            Default: Disables warnings from all modules.

    """
    warnings.filterwarnings("ignore", module=module, category=PossibleUserWarning)


def _custom_format_warning(
    message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str] = None
) -> str:
    """Custom formatting that avoids an extra line in case warnings are emitted from the `rank_zero`-functions."""
    if _is_path_in_lightning(Path(filename)):
        # The warning originates from the Lightning package
        return f"{filename}:{lineno}: {message}\n"
    return _default_format_warning(message, category, filename, lineno, line)


warnings.formatwarning = _custom_format_warning


class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""


def _is_path_in_lightning(path: Path) -> bool:
    """Naive check whether the path looks like a path from the lightning package."""
    return "lightning" in str(path.absolute())
