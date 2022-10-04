import torch.distributed

from lightning_lite import LightningLite

if __name__ == "__main__":
    lite = LightningLite(accelerator="cpu", devices=2, strategy="ddp")
    lite.launch()
    print("launched", lite.global_rank)
    assert torch.distributed.is_initialized()
    lite.barrier()
    print("end")
