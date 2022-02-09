import bagua
import deepspeed
import fairscale
import horovod.torch

print(fairscale.__version__)
print(deepspeed.__version__)
print(bagua.__version__)
print(horovod.torch.nccl_built())
