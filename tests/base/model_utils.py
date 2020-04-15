from torch.utils.data import DataLoader
from tests.base.datasets import TestingMNIST


def dataloader(train, data_root, batch_size):
    dataset = TestingMNIST(root=data_root, train=train, download=False)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader
