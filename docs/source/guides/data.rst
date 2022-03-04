.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from torch.utils.data import IterableDataset, DataLoader, Dataset
    from pytorch_lightning.trainer.trainer import Trainer

.. _data:

#############
Managing Data
#############

****************************
Data Containers in Lightning
****************************

There are a few different data containers used in Lightning:

.. list-table:: Data objects
   :widths: 20 80
   :header-rows: 1

   * - Object
     - Definition
   * - :class:`~torch.utils.data.Dataset`
     - The PyTorch :class:`~torch.utils.data.Dataset` represents a map from keys to data samples.
   * - :class:`~torch.utils.data.IterableDataset`
     - The PyTorch :class:`~torch.utils.data.IterableDataset` represents a stream of data.
   * - :class:`~torch.utils.data.DataLoader`
     - The PyTorch :class:`~torch.utils.data.DataLoader` represents a Python iterable over a Dataset.
   * - :class:`~pytorch_lightning.core.datamodule.LightningDataModule`
     -  A :class:`~pytorch_lightning.core.datamodule.LightningDataModule` is simply a collection of: training DataLoader(s), validation DataLoader(s), test DataLoader(s) and predict DataLoader(s), along with the matching transforms and data processing/downloads steps required.


Why Use LightningDataModule?
============================

The :class:`~pytorch_lightning.core.datamodule.LightningDataModule` was designed as a way of decoupling data-related hooks from the :class:`~pytorch_lightning.core.lightning.LightningModule` so you can develop dataset agnostic models. The :class:`~pytorch_lightning.core.datamodule.LightningDataModule` makes it easy to hot swap different Datasets with your model, so you can test it and benchmark it across domains. It also makes sharing and reusing the exact data splits and transforms across projects possible.

Read :ref:`this <datamodules>` for more details on LightningDataModule.

---------

.. _multiple-dataloaders:

*****************
Multiple Datasets
*****************

There are a few ways to pass multiple Datasets to Lightning:

1. Create a DataLoader that iterates over multiple Datasets under the hood.
2. In the training loop, you can pass multiple DataLoaders as a dict or list/tuple, and Lightning will
   automatically combine the batches from different DataLoaders.
3. In the validation, test, or prediction, you have the option to return multiple DataLoaders as list/tuple, which Lightning will call sequentially
   or combine the DataLoaders using :class:`~pytorch_lightning.trainer.supporters.CombinedLoader`, which Lightning will
   automatically combine the batches from different DataLoaders.


Using LightningDataModule
=========================

You can set more than one :class:`~torch.utils.data.DataLoader` in your :class:`~pytorch_lightning.core.datamodule.LightningDataModule` using its DataLoader hooks
and Lightning will use the correct one.

.. testcode::

    class DataModule(LightningDataModule):

        ...

        def train_dataloader(self):
            return DataLoader(self.train_dataset)

        def val_dataloader(self):
            return [DataLoader(self.val_dataset_1), DataLoader(self.val_dataset_2)]

        def test_dataloader(self):
            return DataLoader(self.test_dataset)

        def predict_dataloader(self):
            return DataLoader(self.predict_dataset)


Using LightningModule Hooks
===========================

Concatenated Dataset
--------------------

For training with multiple Datasets, you can create a :class:`~torch.utils.data.DataLoader` class
which wraps your multiple Datasets using :class:`~torch.utils.data.ConcatDataset`. This, of course,
also works for testing, validation, and prediction Datasets.

.. testcode::

    from torch.utils.data import ConcatDataset


    class LitModel(LightningModule):
        def train_dataloader(self):
            concat_dataset = ConcatDataset(datasets.ImageFolder(traindir_A), datasets.ImageFolder(traindir_B))

            loader = DataLoader(
                concat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
            )
            return loader

        def val_dataloader(self):
            # SAME
            ...

        def test_dataloader(self):
            # SAME
            ...


Return Multiple DataLoaders
---------------------------

You can set multiple DataLoaders in your :class:`~pytorch_lightning.core.lightning.LightningModule`, and Lightning will take care of batch combination.

For more details, refer to :paramref:`~pytorch_lightning.trainer.trainer.Trainer.multiple_trainloader_mode`

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):

            loader_a = DataLoader(range(6), batch_size=4)
            loader_b = DataLoader(range(15), batch_size=5)

            # pass loaders as a dict. This will create batches like this:
            # {'a': batch from loader_a, 'b': batch from loader_b}
            loaders = {"a": loader_a, "b": loader_b}

            # OR:
            # pass loaders as sequence. This will create batches like this:
            # [batch from loader_a, batch from loader_b]
            loaders = [loader_a, loader_b]

            return loaders

Furthermore, Lightning also supports nested lists and dicts (or a combination).

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):

            loader_a = DataLoader(range(8), batch_size=4)
            loader_b = DataLoader(range(16), batch_size=2)

            return {"a": loader_a, "b": loader_b}

        def training_step(self, batch, batch_idx):
            # access a dictionary with a batch from each DataLoader
            batch_a = batch["a"]
            batch_b = batch["b"]


.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):

            loader_a = DataLoader(range(8), batch_size=4)
            loader_b = DataLoader(range(16), batch_size=4)
            loader_c = DataLoader(range(32), batch_size=4)
            loader_c = DataLoader(range(64), batch_size=4)

            # pass loaders as a nested dict. This will create batches like this:
            loaders = {"loaders_a_b": [loader_a, loader_b], "loaders_c_d": {"c": loader_c, "d": loader_d}}
            return loaders

        def training_step(self, batch, batch_idx):
            # access the data
            batch_a_b = batch["loaders_a_b"]
            batch_c_d = batch["loaders_c_d"]

            batch_a = batch_a_b[0]
            batch_b = batch_a_b[1]

            batch_c = batch_c_d["c"]
            batch_d = batch_c_d["d"]

Alternatively, you can also pass in a :class:`~pytorch_lightning.trainer.supporters.CombinedLoader` containing multiple DataLoaders.

.. testcode::

    from pytorch_lightning.trainer.supporters import CombinedLoader


    def train_dataloader(self):
        loader_a = DataLoader()
        loader_b = DataLoader()
        loaders = {"a": loader_a, "b": loader_b}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loader


    def training_step(self, batch, batch_idx):
        batch_a = batch["a"]
        batch_b = batch["b"]


Multiple Validation/Test/Predict DataLoaders
============================================

For validation, test and predict DataLoaders, you can pass a single DataLoader or a list of them. This optional named
parameter can be used in conjunction with any of the above use cases. You can choose to pass
the batches sequentially or simultaneously, as is done for the training step.
The default mode for these DataLoaders is sequential. Note that when using a sequence of DataLoaders you need
to add an additional argument ``dataloader_idx`` in their corresponding step specific hook. The corresponding loop will process
the DataLoaders in sequential order; that is, the first DataLoader will be processed completely, then the second one, and so on.

Refer to the following for more details for the default sequential option:

- :meth:`~pytorch_lightning.core.hooks.DataHooks.val_dataloader`
- :meth:`~pytorch_lightning.core.hooks.DataHooks.test_dataloader`
- :meth:`~pytorch_lightning.core.hooks.DataHooks.predict_dataloader`

.. testcode::

    def val_dataloader(self):
        loader_1 = DataLoader()
        loader_2 = DataLoader()
        return [loader_1, loader_2]


    def validation_step(self, batch, batch_idx, dataloader_idx):
        ...


Evaluation DataLoaders are iterated over sequentially. If you want to iterate over them in parallel, PyTorch Lightning provides a :class:`~pytorch_lightning.trainer.supporters.CombinedLoader` object which supports collections of DataLoaders such as list, tuple, or dictionary. The DataLoaders can be accessed using in the same way as the provided structure:

.. testcode::

    from pytorch_lightning.trainer.supporters import CombinedLoader


    def val_dataloader(self):
        loader_a = DataLoader()
        loader_b = DataLoader()
        loaders = {"a": loader_a, "b": loader_b}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders


    def validation_step(self, batch, batch_idx):
        batch_a = batch["a"]
        batch_b = batch["b"]


Evaluate with Additional DataLoaders
====================================

You can evaluate your models using additional DataLoaders even if the DataLoader specific hooks haven't been defined within your
:class:`~pytorch_lightning.core.lightning.LightningModule`. For example, this would be the case if your test data
set is not available at the time your model was declared. Simply pass the test set to the :meth:`~pytorch_lightning.trainer.trainer.Trainer.test` method:

.. code-block:: python

    # setup your DataLoader
    test = DataLoader(...)

    # test (pass in the loader)
    trainer.test(dataloaders=test)

--------------

********************************************
Accessing DataLoaders within LightningModule
********************************************

In the case that you require access to the DataLoader or Dataset objects, DataLoaders for each step can be accessed using the ``Trainer`` object:

.. testcode::

    from pytorch_lightning import LightningModule


    class Model(LightningModule):
        def test_step(self, batch, batch_idx, dataloader_idx):
            test_dl = self.trainer.test_dataloaders[dataloader_idx]
            test_dataset = test_dl.dataset
            test_sampler = test_dl.sampler
            ...
            # extract metadata, etc. from the dataset:
            ...

If you are using a :class:`~pytorch_lightning.trainer.supporters.CombinedLoader` object which allows you to fetch batches from a collection of DataLoaders
simultaneously which supports collections of DataLoader such as list, tuple, or dictionary. The DataLoaders can be accessed using the same collection structure:

.. code-block:: python

    from pytorch_lightning.trainer.supporters import CombinedLoader

    test_dl1 = ...
    test_dl2 = ...

    # If you provided a list of DataLoaders:

    combined_loader = CombinedLoader([test_dl1, test_dl2])
    list_of_loaders = combined_loader.loaders
    test_dl1 = list_of_loaders.loaders[0]


    # If you provided dictionary of DataLoaders:

    combined_loader = CombinedLoader({"dl1": test_dl1, "dl2": test_dl2})
    dictionary_of_loaders = combined_loader.loaders
    test_dl1 = dictionary_of_loaders["dl1"]

--------------

.. _sequential-data:

***************
Sequential Data
***************

Lightning has built in support for dealing with sequential data.


Packed Sequences as Inputs
==========================

When using :class:`~torch.nn.utils.rnn.PackedSequence`, do two things:

1. Return either a padded tensor in dataset or a list of variable length tensors in the DataLoader's `collate_fn <https://pytorch.org/docs/stable/data.html#dataloader-collate-fn>`_ (example shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

|

.. testcode::

    # For use in DataLoader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y


    # In LightningModule
    def training_step(self, batch, batch_idx):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)


Truncated Backpropagation Through Time (TBPTT)
==============================================

There are times when multiple backwards passes are needed for each batch.
For example, it may save memory to use **Truncated Backpropagation Through Time** when training RNNs.

Lightning can handle TBPTT automatically via this flag.

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
            return {"loss": ..., "hiddens": hiddens}

.. note:: If you need to modify how the batch is split,
    override :func:`~pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch`.


Iterable Datasets
=================
Lightning supports using :class:`~torch.utils.data.IterableDataset` as well as map-style Datasets. IterableDatasets provide a more natural
option when using sequential data.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` you must set the ``val_check_interval`` to 1.0 (the default) or an int
    (specifying the number of training batches to run before each validation loop) when initializing the Trainer. This is
    because the IterableDataset does not have a ``__len__`` and Lightning requires this to calculate the validation
    interval when ``val_check_interval`` is less than one. Similarly, you can set ``limit_{mode}_batches`` to a float or
    an int. If it is set to 0.0 or 0, it will set ``num_{mode}_batches`` to 0, if it is an int, it will set ``num_{mode}_batches``
    to ``limit_{mode}_batches``, if it is set to 1.0 it will run for the whole dataset, otherwise it will throw an exception.
    Here ``mode`` can be train/val/test/predict.

When iterable datasets are used, Lightning will pre-fetch 1 batch (in addition to the current batch) so it can detect
when the training will stop and run validation if necessary.

.. testcode::

    # IterableDataset
    class CustomDataset(IterableDataset):
        def __init__(self, data):
            self.data_source = data

        def __iter__(self):
            return iter(self.data_source)


    # Setup DataLoader
    def train_dataloader(self):
        seq_data = ["A", "long", "time", "ago", "in", "a", "galaxy", "far", "far", "away"]
        iterable_dataset = CustomDataset(seq_data)

        dataloader = DataLoader(dataset=iterable_dataset, batch_size=5)
        return dataloader


.. testcode::

    # Set val_check_interval
    trainer = Trainer(val_check_interval=100)

    # Set limit_val_batches to 0.0 or 0
    trainer = Trainer(limit_val_batches=0.0)

    # Set limit_val_batches as an int
    trainer = Trainer(limit_val_batches=100)
