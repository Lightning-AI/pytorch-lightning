import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer

global_batch_size = 4
micro_batch_size = 2
assert global_batch_size % micro_batch_size == 0


class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.randn(length, 32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.val_fetched = 0
        self.val_iter_raised = False
        self.val_iter_done = False
        self.val_step_entered = 0

        self.train_fetched = 0
        self.train_iter_raised = False
        self.train_iter_done = False
        self.train_step_entered = 0

    def training_step(self, dataloader_iter, batch_idx):
        self.train_step_entered += 1
        self.train_iter_done = dataloader_iter.done
        for i in range(global_batch_size // micro_batch_size):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                self.train_iter_raised = True
                return
            self.train_fetched += 1
        return self.layer(batch).sum()

    def validation_step(self, dataloader_iter, batch_idx):
        self.val_step_entered += 1
        self.val_iter_done = dataloader_iter.done
        for i in range(global_batch_size // micro_batch_size):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                self.val_iter_raised = True
                return
            self.val_fetched += 1
            self.layer(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


train_data = DataLoader(RandomDataset(length=16), batch_size=micro_batch_size)
val_data = DataLoader(RandomDataset(length=16), batch_size=micro_batch_size)

model = BoringModel()
trainer = Trainer(
    # limit_train_batches=3,
    # limit_val_batches=4,
    num_sanity_val_steps=2,
    max_steps=2,
    max_epochs=1,
    accelerator="cpu",
)
trainer.fit(model, train_data, val_data)
# trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

print("train fetched", model.train_fetched)
print("train step entered", model.train_step_entered)
print("train iter exhausted", model.train_iter_raised)

print("val fetched", model.val_fetched)
print("val step entered", model.val_step_entered)
print("val iter exhausted", model.val_iter_raised)
