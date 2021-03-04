import os
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, Trainer
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len
class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)
    def forward(self, x):
        return self.layer(x)
    def training_step(self, batch, batch_idx):
        return self(batch).sum()
    def training_epoch_end(self, outputs) -> None:
        #data = torch.tensor(100)
        data = str(self.global_rank)
        print("before broadcast", self.global_rank, data)
        out = self.trainer.training_type_plugin.broadcast(str(data))
        # data should remain same
        assert data == str(self.global_rank)
        # out is the broadcast result from rank 0
        assert out == "0"
        print(self.global_rank, data, out)
if __name__ == '__main__':
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        gpus=2, # num_processes=2,
        accelerator="ddp",
        fast_dev_run=True
    )
    trainer.fit(model, train_data)