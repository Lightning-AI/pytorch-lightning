from torch.utils.data import DataLoader

from tests.base.datasets import TrialMNIST


class ModelTemplateData:
    hparams: ...

    def dataloader(self, train):
        dataset = TrialMNIST(root=self.hparams.data_root, train=train, download=True)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            # test and valid shall not be shuffled
            shuffle=train,
        )
        return loader


class ModelTemplateUtils:

    def get_output_metric(self, output, name):
        if isinstance(output, dict):
            val = output[name]
        else:  # if it is 2level deep -> per dataloader and per batch
            val = sum(out[name] for out in output) / len(output)
        return val


class CustomInfDataloader:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= 50:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)
