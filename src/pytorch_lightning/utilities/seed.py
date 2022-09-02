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
"""Utilities to help with reproducibility of models."""

import logging
from contextlib import contextmanager
from typing import Generator

import torch

# For backward-compatibility
# TODO(lite): deprecate usage
from lightning_lite.utilities.seed import (  # noqa: F401
    _collect_rng_states,
    _set_rng_states,
    pl_worker_init_function,
    reset_seed,
    seed_everything,
)

log = logging.getLogger(__name__)


@contextmanager
def isolate_rng() -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Example:
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])
    """
    states = _collect_rng_states()
    yield
    _set_rng_states(states)
