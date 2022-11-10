from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12

if _TORCH_GREATER_EQUAL_1_12:
    # TODO: Remove guard once we support testing bagua on >= 1.13
    import bagua  # noqa: F401

import deepspeed  # noqa: F401
import fairscale  # noqa: F401
import horovod.torch

# returns an error code
assert horovod.torch.nccl_built()
