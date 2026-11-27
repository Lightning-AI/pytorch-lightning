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
"""Utilities that can be used for calling functions on a particular rank."""

import logging
from functools import wraps
from typing import Any

from lightning.fabric.utilities import rank_zero as _fabric_rank_zero

# note: we want to keep these indirections so the `rank_zero_module.log` is set (on import) for PL users
LightningDeprecationWarning = _fabric_rank_zero.LightningDeprecationWarning
WarningCache = _fabric_rank_zero.WarningCache
rank_prefixed_message = _fabric_rank_zero.rank_prefixed_message
rank_zero_deprecation = _fabric_rank_zero.rank_zero_deprecation
rank_zero_module = _fabric_rank_zero.rank_zero_module
rank_zero_only = _fabric_rank_zero.rank_zero_only


def _set_rank_zero_logger() -> None:
    rank_zero_module.log = logging.getLogger(__name__)


@wraps(_fabric_rank_zero.rank_zero_debug)
def rank_zero_debug(*args: Any, **kwargs: Any) -> None:
    _set_rank_zero_logger()
    return _fabric_rank_zero.rank_zero_debug(*args, **kwargs)


@wraps(_fabric_rank_zero.rank_zero_info)
def rank_zero_info(*args: Any, **kwargs: Any) -> None:
    _set_rank_zero_logger()
    return _fabric_rank_zero.rank_zero_info(*args, **kwargs)


@wraps(_fabric_rank_zero.rank_zero_warn)
def rank_zero_warn(*args: Any, **kwargs: Any) -> None:
    _set_rank_zero_logger()
    return _fabric_rank_zero.rank_zero_warn(*args, **kwargs)


_set_rank_zero_logger()
