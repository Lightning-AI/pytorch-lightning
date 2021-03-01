import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
"""
This will generate 2 traces: one for `training_step` and one for `validation_step`.
Open Chrome and copy/paste this url: `chrome://tracing/`.
Once tracing opens, click on `Load` at the top-right and load one of the generated traces.
"""


class LitLightningModule(LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class CIFAR10DataModule(LightningDataModule):

    def train_dataloader(self, *args, **kwargs):
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    def val_dataloader(self, *args, **kwargs):
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=0)


if __name__ == '__main__':

    model = LitLightningModule(models.resnet50(pretrained=True))
    datamodule = CIFAR10DataModule()
    trainer = Trainer(max_epochs=1, limit_train_batches=15, limit_val_batches=15, profiler="pytorch")
    trainer.fit(model, datamodule=datamodule)
