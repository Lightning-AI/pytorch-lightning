.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule

.. _multiple_loaders:

Multiple Datasets
=================
Lightning supports multiple dataloaders in a few ways.

1. Create a dataloader that iterates multiple datasets under the hood.
2. In the training loop you can pass multiple loaders as a dict or list/tuple and lightning
   will automatically combine the batches from different loaders.
3. In the validation and test loop you also have the option to return multiple dataloaders
   which lightning will call sequentially.

----------

.. _multiple-training-dataloaders:

Multiple training dataloaders
-----------------------------
For training, the usual way to use multiple dataloaders is to create a ``DataLoader`` class
which wraps your multiple dataloaders (this of course also works for testing and validation
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

However, with lightning you can also return multiple loaders and lightning will take care of batch combination.

For more details please have a look at :paramref:`~pytorch_lightning.trainer.trainer.Trainer.multiple_trainloader_mode`

.. testcode::

    class LitModel(LightningModule):

        def train_dataloader(self):

            loader_a = torch.utils.data.DataLoader(range(6), batch_size=4)
            loader_b = torch.utils.data.DataLoader(range(15), batch_size=5)

            # pass loaders as a dict. This will create batches like this:
            # {'a': batch from loader_a, 'b': batch from loader_b}
            loaders = {'a': loader_a,
                       'b': loader_b}

            # OR:
            # pass loaders as sequence. This will create batches like this:
            # [batch from loader_a, batch from loader_b]
            loaders = [loader_a, loader_b]

            return loaders

Furthermore, Lightning also supports that nested lists and dicts (or a combination) can
be returned.

.. testcode::

    class LitModel(LightningModule):

        def train_dataloader(self):

            loader_a = torch.utils.data.DataLoader(range(8), batch_size=4)
            loader_b = torch.utils.data.DataLoader(range(16), batch_size=2)

            return {'a': loader_a, 'b': loader_b}

        def training_step(self, batch, batch_idx):
            # access a dictionnary with a batch from each dataloader
            batch_a = batch["a"]
            batch_b = batch["b"]


.. testcode::

    class LitModel(LightningModule):

        def train_dataloader(self):

            loader_a = torch.utils.data.DataLoader(range(8), batch_size=4)
            loader_b = torch.utils.data.DataLoader(range(16), batch_size=4)
            loader_c = torch.utils.data.DataLoader(range(32), batch_size=4)
            loader_c = torch.utils.data.DataLoader(range(64), batch_size=4)

            # pass loaders as a nested dict. This will create batches like this:
            loaders = {
                'loaders_a_b': {
                    'a': loader_a,
                    'b': loader_b
                },
                'loaders_c_d': {
                    'c': loader_c,
                    'd': loader_d
                }
            }
            return loaders

        def training_step(self, batch, batch_idx):
            # access the data
            batch_a_b = batch["loaders_a_b"]
            batch_c_d = batch["loaders_c_d"]

            batch_a = batch_a_b["a"]
            batch_b = batch_a_b["a"]

            batch_c = batch_c_d["c"]
            batch_d = batch_c_d["d"]

----------

Test/Val dataloaders
--------------------
For validation and test dataloaders, lightning also gives you the additional
option of passing multiple dataloaders back from each call. You can choose to pass
the batches sequentially or simultaneously, as is done for the training step.
The default mode for validation and test dataloaders is sequential.

See the following for more details for the default sequential option:

- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.test_dataloader`

.. testcode::

    def val_dataloader(self):
        loader_1 = Dataloader()
        loader_2 = Dataloader()
        return [loader_1, loader_2]

To combine batches of multiple test and validation dataloaders simultaneously, one
needs to wrap the dataloaders with `CombinedLoader`.

.. testcode::

    from pytorch_lightning.trainer.supporters import CombinedLoader

    def val_dataloader(self):
        loader_1 = Dataloader()
        loader_2 = Dataloader()
        loaders = {'a': loader_a,'b': loader_b}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders
