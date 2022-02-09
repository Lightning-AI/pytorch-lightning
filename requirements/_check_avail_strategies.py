import bagua
import deepspeed
import fairscale
import horovod.torch  # noqa: F401

print(fairscale.__version__)
print(deepspeed.__version__)
print(bagua.__version__)
