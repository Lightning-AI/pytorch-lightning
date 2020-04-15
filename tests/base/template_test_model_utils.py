from torch.utils.data import DataLoader
from tests.base.datasets import TestingMNIST


class TemplateTestModelUtilsMixin:

    def dataloader(self, train, data_root, batch_size):
        dataset = TestingMNIST(root=data_root, train=train, download=False)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return loader

    def get_output_metric(self,  output, name):
        if isinstance(output, dict):
            val = output[name]
        else:  # if it is 2level deep -> per dataloader and per batch
            val = sum(out[name] for out in output) / len(output)
        return val
