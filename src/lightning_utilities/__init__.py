"""Root package info."""

import os

from lightning_utilities.__about__ import *  # noqa: F403
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.enums import StrEnum
from lightning_utilities.core.imports import compare_version, module_available
from lightning_utilities.core.overrides import is_overridden
from lightning_utilities.core.rank_zero import WarningCache

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


__all__ = [
    "apply_to_collection",
    "StrEnum",
    "module_available",
    "compare_version",
    "is_overridden",
    "WarningCache",
]
