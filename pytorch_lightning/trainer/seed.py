"""Helper functions to help with reproducibility of models """
import os

import numpy as np
import random
import torch

from pytorch_lightning import _logger as log


def seed_everything(seed: int = None):
    """Function that sets seed for pseudo-random number generators  in:
        pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(0, max_seed_value)
        log.info(f"No seed found, seed set to {seed}")

    assert seed <= max_seed_value, "seed is too big, numpy accepts only uint32"
    assert seed >= min_seed_value, "seed is too small, numpy accepts only uint32"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
