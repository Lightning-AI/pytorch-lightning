.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule

Multiple Datasets
=================
Lightning supports multiple dataloaders in a few ways.

1. Create a dataloader that iterates both datasets under the hood.
2. In the validation and test loop you also have the option to return multiple dataloaders
   which lightning will call sequentially.

Multiple training dataloaders
-----------------------------
For training, the best way to use multiple-dataloaders is to create a Dataloader class
which wraps both your dataloaders. (This of course also works for testing and validation
dataloaders).

(`reference <https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2>`_)

.. testcode::

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)

    class LitModel(LightningModule):

        def train_dataloader(self):
            concat_dataset = ConcatDataset(
                datasets.ImageFolder(traindir_A),
                datasets.ImageFolder(traindir_B)
            )

            loader = torch.utils.data.DataLoader(
                concat_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )
            return loader

        def val_dataloader(self):
            # SAME
            ...

        def test_dataloader(self):
            # SAME
            ...

Test/Val dataloaders
--------------------
For validation, test dataloaders lightning also gives you the additional
option of passing in multiple dataloaders back from each call.

See the following for more details:

- :meth:`~pytorch_lightning.core.LightningModule.val_dataloader`
- :meth:`~pytorch_lightning.core.LightningModule.test_dataloader`

.. testcode::

    def val_dataloader(self):
        loader_1 = Dataloader()
        loader_2 = Dataloader()
        return [loader_1, loader_2]
