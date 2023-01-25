# Copyright The Lightning team.
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
from contextlib import contextmanager
from typing import Any, Generator

from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_fabric.utilities.seed import pl_worker_init_function as new_pl_worker_init_function
from lightning_fabric.utilities.seed import reset_seed as new_reset_seed
from lightning_fabric.utilities.seed import seed_everything as new_seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


@contextmanager
def isolate_rng() -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Example:
        >>> import torch
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


def seed_everything(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.seed.seed_everything` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.seed.seed_everything` instead."
    )
    return new_seed_everything(*args, **kwargs)


def reset_seed() -> None:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.seed.reset_seed` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.seed.reset_seed` instead."
    )
    return new_reset_seed()


def pl_worker_init_function(*args: Any, **kwargs: Any) -> None:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.seed.pl_worker_init_function` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.seed.pl_worker_init_function` instead."
    )
    return new_pl_worker_init_function(*args, **kwargs)
