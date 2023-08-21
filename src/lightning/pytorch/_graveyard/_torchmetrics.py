import contextlib
from typing import Callable

import torchmetrics
from lightning_utilities.core.imports import compare_version as _compare_version

from lightning.pytorch.utilities.migration.utils import _patch_pl_to_mirror_if_necessary


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    new_package = _patch_pl_to_mirror_if_necessary(package)
    return _compare_version(new_package, op, version, use_base_version)


# patching is necessary, since up to v.0.7.3 torchmetrics has a hardcoded reference to lightning.pytorch,
# which has to be redirected to the unified package:
# https://github.com/Lightning-AI/metrics/blob/v0.7.3/torchmetrics/metric.py#L96
with contextlib.suppress(AttributeError):
    if hasattr(torchmetrics.utilities.imports, "_compare_version"):
        torchmetrics.utilities.imports._compare_version = compare_version

with contextlib.suppress(AttributeError):
    if hasattr(torchmetrics.metric, "_compare_version"):
        torchmetrics.metric._compare_version = compare_version
