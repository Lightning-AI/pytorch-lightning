Multiple Dataloaders
====================
Lightning supports multiple dataloaders in a few ways.

Multiple training dataloaders
-----------------------------
For training, the best way to use multiple-dataloaders is to create a Dataloader class
which wraps both your dataloaders. (This of course also works for testing and validation
dataloaders).

(`reference <https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2>`_

.. code-block::

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)

        concat_dataset = ConcatDataset(
            datasets.ImageFolder(traindir_A),
            datasets.ImageFolder(traindir_B)
        )

    class LitModel(LightningModule):
        def train_dataloader(self):
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

        def test_dataloader(self):
            # SAME

Test/Val dataloaders
--------------------
For validation, test dataloaders lightning also gives you the additional
option of passing in multiple dataloaders back from each call.

See the following for more details:

- :meth:`~pytorch_lightning.core.LightningModule.val_dataloader`
- :meth:`~pytorch_lightning.core.LightningModule.test_dataloader`

.. code-block::

    def val_dataloader(self):
        loader_1 = Dataloader()
        loader_2 = Dataloader()
        return [loader_1, loader_2]
