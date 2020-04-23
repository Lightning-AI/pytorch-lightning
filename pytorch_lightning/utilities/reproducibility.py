"""Helper functions to help with reproducibility of models """
import os

import numpy as np
import random
import torch


def seed_everything(seed: int = None):
    """Function that sets seed for pseudo-random number generators  """
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
