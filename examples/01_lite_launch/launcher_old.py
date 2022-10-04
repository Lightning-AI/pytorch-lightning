import torch.distributed

from lightning_lite import LightningLite


class Lite(LightningLite):
    def run(self):
        print("launched", self.global_rank)
        assert torch.distributed.is_initialized()
        self.barrier()


if __name__ == "__main__":
    lite = Lite(accelerator="cpu", devices=2, strategy="ddp")
    lite.run()
    print("after run", lite.global_rank)
