
.. testsetup:: *

    from pytorch_lightning.core.Lightning import LightningModule
    from torch.utils.data import IterableDataSet
    from pytorch_lightning.trainer.trainer import Trainer

.. _data:

#############
Managing Data
#############

There are a few different Data objects used in Lightning:

.. list-table:: Data objects
   :widths: 20 80
   :header-rows: 1

   * - Object
     - Definition
   * - :class:`~torch.utils.data.Dataset`
     - The PyTorch :class:`~torch.utils.data.Dataset` represents a map from keys to data samples.
   * - :class:`~torch.utils.data.DataLoader`
     - The PyTorch :class:`~torch.utils.data.DataLoader` represents a Python iterable over a DataSet.
   * - :class:`~pytorch_lightning.core.datamodule.LightningDataModule`
     - A :class:`~pytorch_lightning.core.datamodule.LightningDataModule` is simply a collection of a training DataLoader, validation DataLoader(s) and test DataLoader(s), along with the matching transforms and data processing/downloads steps required.

See :ref:`here <datamodules>` more details on LightningDataModules.

In this guide you can find out how to:

* `Iterate over multiple datasets <Multiple DataSets_>`_

* `Handle sequential data <Sequential Data_>`_


.. _multiple-training-dataloaders:

*****************
Multiple DataSets
*****************

There are a few ways to pass multiple DataSets to Lightning:

1. Create a DataLoader that iterates over multiple DataSets under the hood.
2. In the training loop you can pass multiple DataLoaders as a dict or list/tuple and Lightning
   will automatically combine the batches from different DataLoaders.
3. In the validation and test loop you have the option to return multiple DataLoaders,
   which Lightning will call sequentially.


Using LightningDataModule
=========================
You can set multiple DataLoaders in your :class:`~pytorch_lightning.core.datamodule.LightningDataModule`, and Lightning will handle the
combination batch under-the-hood.

TODO: add code snippet.

Using LightningModule hooks
===========================

Concatenated DataSet
--------------------
For training with multiple datasets you can create a :class:`~torch.utils.data.dataloader` class
which wraps your multiple DataSets (this of course also works for testing and validation
DataSets).

(`reference <https://discuss.pytorch.org/t/train-simultaneously-on-two-DataSets/649/2>`_)

.. testcode::

    class ConcatDataSet(torch.utils.data.DataSet):
        def __init__(self, *DataSets):
            self.DataSets = DataSets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.DataSets)

        def __len__(self):
            return min(len(d) for d in self.DataSets)

    class LitModel(LightningModule):

        def train_dataloader(self):
            concat_DataSet = ConcatDataSet(
                DataSets.ImageFolder(traindir_A),
                DataSets.ImageFolder(traindir_B)
            )

            loader = torch.utils.data.DataLoader(
                concat_DataSet,
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

Return multiple DataLoaders
---------------------------
You can set multiple DataLoaders in your :class:`~pytorch_lightning.core.lightning.LightningModule`, and Lightning will take care of batch combination.

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

Furthermore, Lightning also supports nested lists and dicts (or a combination).

.. testcode::

    class LitModel(LightningModule):

        def train_dataloader(self):

            loader_a = torch.utils.data.DataLoader(range(8), batch_size=4)
            loader_b = torch.utils.data.DataLoader(range(16), batch_size=2)

            return {'a': loader_a, 'b': loader_b}

        def training_step(self, batch, batch_idx):
            # access a dictionnary with a batch from each DataLoader
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

Multiple validation/test datasets
=================================
For validation and test DataLoaders, you can pass a single DataLoader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases. You can choose to pass
the batches sequentially or simultaneously, as is done for the training step.
The default mode for validation and test DataLoaders is sequential.

See the following for more details for the default sequential option:

- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader`
- :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.test_dataloader`

.. testcode::

    def val_dataloader(self):
        loader_1 = DataLoader()
        loader_2 = DataLoader()
        return [loader_1, loader_2]

To combine batches of multiple test and validation DataLoaders simultaneously, one
needs to wrap the DataLoaders with `CombinedLoader`.

.. testcode::

    from pytorch_lightning.trainer.supporters import CombinedLoader

    def val_dataloader(self):
        loader_1 = DataLoader()
        loader_2 = DataLoader()
        loaders = {'a': loader_a,'b': loader_b}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders


Test with additional data loaders
=================================
You can run inference on a test set even if the :func:`~pytorch_lightning.core.Lightning.LightningModule.test_dataloader` method hasn't been
defined within your :class:`~pytorch_lightning.core.Lightning.LightningModule` instance. For example, rhis would be the case if your test data
set is not available at the time your model was declared. Simply pass the test set to the :func:`~pytorch_lightning.trainer.trainer.Trainer.test` method:

.. code-block:: python

    # setup your data loader
    test = DataLoader(...)

    # test (pass in the loader)
    trainer.test(test_dataloaders=test)

--------------


.. _sequences:


***************
Sequential Data
***************

Lightning has built in support for dealing with sequential data.


Packed sequences as inputs
==========================
When using PackedSequence, do 2 things:

1. Return either a padded tensor in DataSet or a list of variable length tensors in the DataLoader collate_fn (example shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. testcode::

    # For use in DataLoader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y

    # In module
    def training_step(self, batch, batch_nb):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)

----------

Truncated Backpropagation Through Time
======================================
There are times when multiple backwards passes are needed for each batch.
For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

Lightning can handle TBTT automatically via this flag.

.. testcode::

    from pytorch_lightning import LightningModule

    class MyModel(LightningModule):

        def __init__(self):
            super().__init__()
            # Important: This property activates truncated backpropagation through time
            # Setting this value to 2 splits the batch into sequences of size 2
            self.truncated_bptt_steps = 2

        # Truncated back-propagation through time
        def training_step(self, batch, batch_idx, hiddens):
            # the training step must be updated to accept a ``hiddens`` argument
            # hiddens are the hiddens from the previous truncated backprop step
            out, hiddens = self.lstm(data, hiddens)
            return {
                "loss": ...,
                "hiddens": hiddens
            }

.. note:: If you need to modify how the batch is split,
    override :func:`~pytorch_lightning.core.LightningModule.tbptt_split_batch`.

----------

Iterable DataSets
=================
Lightning supports using IterableDataSets as well as map-style DataSets. IterableDataSets provide a more natural
option when using sequential data.

.. note:: When using an IterableDataSet you must set the ``val_check_interval`` to 1.0 (the default) or an int
    (specifying the number of training batches to run before validation) when initializing the Trainer. This is
    because the IterableDataSet does not have a ``__len__`` and Lightning requires this to calculate the validation
    interval when ``val_check_interval`` is less than one. Similarly, you can set ``limit_{mode}_batches`` to a float or
    an int. If it is set to 0.0 or 0 it will set ``num_{mode}_batches`` to 0, if it is an int it will set ``num_{mode}_batches``
    to ``limit_{mode}_batches``, if it is set to 1.0 it will run for the whole DataSet, otherwise it will throw an exception.
    Here mode can be train/val/test.

.. testcode::

    # IterableDataSet
    class CustomDataSet(IterableDataSet):

        def __init__(self, data):
            self.data_source

        def __iter__(self):
            return iter(self.data_source)

    # Setup DataLoader
    def train_dataloader(self):
        seq_data = ['A', 'long', 'time', 'ago', 'in', 'a', 'galaxy', 'far', 'far', 'away']
        iterable_DataSet = CustomDataSet(seq_data)

        dataloader = DataLoader(DataSet=iterable_DataSet, batch_size=5)
        return dataloader

.. testcode::

    # Set val_check_interval
    trainer = Trainer(val_check_interval=100)

    # Set limit_val_batches to 0.0 or 0
    trainer = Trainer(limit_val_batches=0.0)

    # Set limit_val_batches as an int
    trainer = Trainer(limit_val_batches=100)
