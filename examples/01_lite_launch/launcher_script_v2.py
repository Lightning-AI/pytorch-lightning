import torch.distributed

from lightning_lite import LightningLite


def run(lite):
    print("launched", lite.global_rank)
    assert torch.distributed.is_initialized()
    lite.barrier()
    print("end")


if __name__ == "__main__":
    lite = LightningLite(accelerator="cpu", devices=2, strategy="ddp")
    lite.launch(run)
