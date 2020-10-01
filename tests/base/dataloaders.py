"""Custom dataloaders for testing"""


class CustomInfDataloader:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.count = 0
        self.dataloader.num_workers = 0  # reduce chance for hanging pytest

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


class CustomNotImplementedErrorDataloader(CustomInfDataloader):

    def __len__(self):
        """raise NotImplementedError"""
        raise NotImplementedError

    def __next__(self):
        if self.count >= 2:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)
