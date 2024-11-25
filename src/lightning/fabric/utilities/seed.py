import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Optional

import torch

from lightning.fabric.utilities.imports import _NUMPY_AVAILABLE
from lightning.fabric.utilities.rank_zero import _get_rank, rank_prefixed_message, rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)


max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0


def seed_everything(seed: Optional[int] = None, workers: bool = False, verbose: bool = True) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.
        verbose: Whether to print a message on each rank with the seed being set.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = 0

    if verbose:
        log.info(rank_prefixed_message(f"Seed set to {seed}", _get_rank()))

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    if _NUMPY_AVAILABLE:
        import numpy as np

        np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def reset_seed() -> None:
    r"""Reset the seed to the value that :func:`~lightning.fabric.utilities.seed.seed_everything` previously set.

    If :func:`~lightning.fabric.utilities.seed.seed_everything` is unused, this function will do nothing.

    """
    seed = os.environ.get("PL_GLOBAL_SEED", None)
    if seed is None:
        return
    workers = os.environ.get("PL_SEED_WORKERS", "0")
    seed_everything(int(seed), workers=bool(int(workers)), verbose=False)


def pl_worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
    r"""The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.

    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    log.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    seed_sequence = _generate_seed_sequence(base_seed, worker_id, global_rank, count=4)
    torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
    random.seed((seed_sequence[1] << 32) | seed_sequence[2])  # combine two 64-bit seeds
    if _NUMPY_AVAILABLE:
        import numpy as np

        ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
        np_rng_seed = ss.generate_state(4)

        np.random.seed(np_rng_seed)


def _generate_seed_sequence(base_seed: int, worker_id: int, global_rank: int, count: int) -> list[int]:
    """Generates a sequence of seeds from a base seed, worker id and rank using the linear congruential generator (LCG)
    algorithm."""
    # Combine base seed, worker id and rank into a unique 64-bit number
    combined_seed = (base_seed << 32) | (worker_id << 16) | global_rank
    seeds = []
    for _ in range(count):
        # x_(n+1) = (a * x_n + c) mod m. With c=1, m=2^64 and a is D. Knuth's constant
        combined_seed = (combined_seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        seeds.append(combined_seed)
    return seeds


def _collect_rng_states(include_cuda: bool = True) -> dict[str, Any]:
    r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "python": python_get_rng_state(),
    }
    if _NUMPY_AVAILABLE:
        import numpy as np

        states["numpy"] = np.random.get_state()
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states


def _set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    r"""Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    if _NUMPY_AVAILABLE and "numpy" in rng_state_dict:
        import numpy as np

        np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))
