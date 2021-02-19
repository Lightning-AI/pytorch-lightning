import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.profiler import PyTorchProfiler


class LitLightningModule(LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion =  torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

class CIFAR10DataModule(LightningDataModule):

    def train_dataloader(self, *args, **kwargs):
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

model = LitLightningModule(models.resnet50(pretrained=True))
datamodule = CIFAR10DataModule()

trainer = Trainer(
    max_epochs=1, 
    limit_train_batches=15,
    gpus=1, 
    profiler="pytorch"
)
trainer.fit(model, datamodule=datamodule)