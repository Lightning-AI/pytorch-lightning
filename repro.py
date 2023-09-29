import torch, timm
import lightning.pytorch as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import CIFAR10
import os


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("rexnet_150", pretrained=True, num_classes=10)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def training_step(self, batch):
        print(os.environ.get("OMP_NUM_THREADS"))
        print(torch.get_num_threads(), torch.get_num_interop_threads())
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss


def run():
    transform = tfs.Compose([tfs.Resize((224, 224)), tfs.ToTensor()])
    dataset = CIFAR10(".", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)
    model = LitModel()
    trainer = pl.Trainer(accelerator="cuda", devices=2, strategy="ddp")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    # torch.set_num_interop_threads(os.cpu_count() // 2)
    # torch.set_num_threads(1)
    run()
