import bagua  # noqa: F401
import deepspeed  # noqa: F401
import fairscale  # noqa: F401
import horovod.torch

# returns an error code
assert horovod.torch.nccl_built()
