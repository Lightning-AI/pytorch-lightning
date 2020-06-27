"""General utilities"""

import torch

from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.parsing import AttributeDict

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

NATIVE_AMP_AVALAIBLE = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
