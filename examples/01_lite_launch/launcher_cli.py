# Run this with
#
# python -m lightning_lite.cli examples/01_lite_launch/launcher_cli.py --devices 2 --precision bf16

import torch.distributed

from lightning_lite import LightningLite


if __name__ == "__main__":
    lite = LightningLite()
    print("launched", lite.global_rank)
    assert torch.distributed.is_initialized()
    lite.barrier()
    print("end")
