from typing import Callable

import torchmetrics
from lightning_utilities.core.imports import compare_version as _compare_version

from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_8_0
from lightning.pytorch.utilities.migration.utils import _patch_pl_to_mirror_if_necessary


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    new_package = _patch_pl_to_mirror_if_necessary(package)
    return _compare_version(new_package, op, version, use_base_version)


if not _TORCHMETRICS_GREATER_EQUAL_0_8_0:
    # up to v0.8.0 torchmetrics had a hardcoded reference to lightning.pytorch which has to be redirected to the
    # unified package. this was removed in
    # https://github.com/Lightning-AI/torchmetrics/commit/b225889b34b83272117b758cbc28772a5c2356d9
    torchmetrics.utilities.imports._compare_version = compare_version
    torchmetrics.metric._compare_version = compare_version
