import torch
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.loops.base import Loop


class TrainingBatchLoop(Loop):
    def __init__(self, model, optimizer, dataloader):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.current_batch_idx = 0

    @property
    def done(self):
        return self.current_batch_idx >= len(self.dataloader)

    def reset(self) -> None:
        self.dataloader_iter = iter(self.dataloader)

    def advance(self, *args, **kwargs) -> None:
        batch = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        loss = self.model(batch)  # , self.current_batch_idx)
        loss.backward()
        self.optimizer.step()


class EpochLoop(Loop):
    def __init__(self, num_epochs, model, optimizer, dataloader):
        super().__init__()
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.training_loop = TrainingBatchLoop(model, optimizer, dataloader)

    @property
    def done(self):
        return self.num_epochs < self.current_epoch

    def reset(self) -> None:
        pass

    def advance(self, *args, **kwargs) -> None:
        self.training_loop.run()
        self.current_epoch += 1


model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
dataloader = DataLoader(torch.zeros((10,)))
EpochLoop(10, model, optimizer, dataloader).run()
