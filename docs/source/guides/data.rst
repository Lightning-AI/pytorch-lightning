
.. testsetup:: *

    from pytorch_lightning.core.Lightning import LightningModule
    from torch.utils.data import IterableDataSet
    from pytorch_lightning.trainer.trainer.Trainer import Trainer

.. _data:

#############
Managing Data
#############

Continue reading to learn about:

* `Data Objects in Lightning <Data Objects in Lightning_>`_

* `Dynamically Configuring Models using Data <Dynamically Configuring Models using Data_>`_

* `Multiple Datasets <Multiple DataSets_>`_

* `Sequential Data <Sequential Data_>`_

*************************
Data Objects in Lightning
*************************

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
   * - :class:`torch.utils.data.IterableDataset`
     - The PyTorch :class:`~torch.utils.data.IterableDataset` represents an iterable over data samples, useful for data streams.
   * - :class:`~pytorch_lightning.core.datamodule.LightningDataModule`
     - A :class:`~pytorch_lightning.core.datamodule.LightningDataModule` is simply a collection of a training DataLoader, validation DataLoader(s) and test DataLoader(s), along with the matching transforms and data processing/downloads steps required.

Creating DataLoaders
====================

To pass in data to Lightning, you'll first need to create a :class:`~torch.utils.data.DataLoader` from a :class:`~torch.utils.data.Dataset`.

For example, here's the PyTorch code for loading an MNIST :class:`~torch.utils.data.DataLoader`:

.. testcode::
    :skipif: not _TORCHVISION_AVAILABLE

    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    import tempfile
    from torchvision import datasets, transforms

    # transforms
    # prepare transforms standard to MNIST
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

    # data
    mnist_train = MNIST(tempfile.mkdtemp(), train=True, download=True, transform=transform)
    mnist_train = DataLoader(mnist_train, batch_size=64)

.. testoutput::
    :hide:
    :skipif: os.path.isdir(os.path.join(os.getcwd(), 'MNIST')) or not _TORCHVISION_AVAILABLE

    Downloading ...
    Extracting ...
    Downloading ...
    Extracting ...
    Downloading ...
    Extracting ...
    Processing...
    Done!

Using DataLoaders in Lightning
==============================

You can pass DataLoaders to the Lightning Trainer in 3 ways:

1. Pass DataLoaders to Trainer.fit()
------------------------------------
Pass in the dataloaders to the :meth:`pytorch_lightning.trainer.trainer.Trainer.fit` function.

.. code-block:: python

    model = LitMNIST()
    trainer = Trainer()
    trainer.fit(model, mnist_train)


2. Pass DataLoaders to the LightningModule
------------------------------------------
For fast research prototyping, it might be easier to add the DataLoaders to your :class:`~pytorch_lightning.core.lightning.LightningModule`, using the DataLoader hooks (:meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader`, :meth:`~pytorch_lightning.core.lightning.LightningModule.val_dataloader`, :meth:`~pytorch_lightning.core.lightning.LightningModule.test_dataloader`).


.. code-block:: python

    class LitMNIST(pl.LightningModule):

        def train_dataloader(self):
            transforms = ...
            mnist_train = ...
            return DataLoader(mnist_train, batch_size=64)

        def val_dataloader(self):
            transforms = ...
            mnist_val = ...
            return DataLoader(mnist_val, batch_size=64)

        def test_dataloader(self):
            transforms = ...
            mnist_test = ...
            return DataLoader(mnist_test, batch_size=64)

The :class:`~pytorch_lightning.core.lightning.LightningModule` contains the DataLoaders, so there's no need to specify on :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`.

.. code-block:: python

    model = LitMNIST()
    trainer = Trainer()
    trainer.fit(model)

3. LightningDataModules (recommended)
-------------------------------------
Defining free-floating dataloaders, splits, download instructions, and such can get messy.
We recommend grouping the full definition of the datasets into a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` which includes:

- Download instructions
- Processing instructions
- Split instructions
- Train dataloader
- Val dataloader(s)
- Test dataloader(s)

.. testcode::

    class MyDataModule(LightningDataModule):

        def __init__(self):
            super().__init__()
            self.train_dims = None
            self.vocab_size = 0

        def prepare_data(self):
            # called only on rank 0 process
            download_dataset()
            tokenize()
            build_vocab()

        def setup(self, stage: Optional[str] = None):
            # called on every process
            vocab = load_vocab()
            self.vocab_size = len(vocab)

            self.train, self.val, self.test = load_datasets()
            self.train_dims = self.train.next_batch.size()

        def train_dataloader(self):
            transforms = ...
            return DataLoader(self.train, batch_size=64)

        def val_dataloader(self):
            transforms = ...
            return DataLoader(self.val, batch_size=64)

        def test_dataloader(self):
            transforms = ...
            return DataLoader(self.test, batch_size=64)
            
        def predict_dataloader(self):
            transforms = ...
            return DataLoader(self.predict, batch_size=64)

DataModules are easier to re-use compared to pure :class:`~torch.utils.data.Dataset` definitions.

.. code-block:: python

    # use an MNIST dataset
    mnist_dm = MNISTDatamodule()
    model = LitModel(num_classes=mnist_dm.num_classes)
    trainer.fit(model, mnist_dm)

    # or other datasets with the same model
    imagenet_dm = ImagenetDatamodule()
    model = LitModel(num_classes=imagenet_dm.num_classes)
    trainer.fit(model, imagenet_dm)

.. note:: :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.prepare_data` is called on only one GPU in distributed training (automatically)
.. note:: :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.setup` is called on every GPU (automatically)

Read :ref:`this <datamodules>` for more details on LightningDataModules.

---------------

*****************************************
Dynamically Configuring Models using Data
*****************************************

When your models need to know about the data, it's best to process the data before passing it to the model.

.. code-block:: python

    # init dm AND call the processing manually
    dm = ImagenetDataModule()
    dm.prepare_data()
    dm.setup()

    model = LitModel(out_features=dm.num_classes, img_width=dm.img_width, img_height=dm.img_height)
    trainer.fit(model, dm)


1. use :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.prepare_data` to download and process the :class:`~torch.utils.data.Dataset`.
2. use :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.setup` to do splits, and build your model internals

An alternative to using a DataModule is to defer initialization of the models modules to the :meth:`~pytorch_lightning.core.datamodule.LightningDataModule.setup` method of your LightningModule as follows:

.. testcode::

    class LitMNIST(LightningModule):

        def __init__(self):
            self.l1 = None

        def prepare_data(self):
            download_data()
            tokenize()

        def setup(self, stage: Optional[str] = None):
            # step is either 'fit', 'validate', 'test', or 'predict'. 90% of the time not relevant
            data = load_data()
            num_classes = data.classes
            self.l1 = nn.Linear(..., num_classes)

--------------

.. _multiple-training-dataloaders:

*****************
Multiple DataSets
*****************

There are a few ways to pass multiple DataSets to Lightning:

1. Create a :class:`~torch.utils.data.DataLoader` that iterates over multiple DataSets under the hood.
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

Concatenated Dataset
--------------------
For training with multiple Datasets you can create a :class:`~torch.utils.data.DataLoader` class
which wraps your multiple DataSets (this of course also works for testing and validation
Datasets).

(`reference <https://discuss.pytorch.org/t/train-simultaneously-on-two-DataSets/649/2>`_)

.. testcode::

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datadets)

        def __len__(self):
            return min(len(d) for d in self.datadets)

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

Return multiple DataLoaders
---------------------------
You can set multiple DataLoaders in your :class:`~pytorch_lightning.core.lightning.LightningModule`, and Lightning will take care of batch combination.

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
            
        def training_step(self, batch, batch_idx):
             # access a list with a batch from each DataLoader
             batch_a, batch_b = batch

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

By default, the trainer ends one epoch when the largest dataset is traversed, and smaller datasets reload when running out of their data.
To change this behaviour, you can set the trainer Flag ``multiple_trainloader_mode=min_size`` to make all the datasets reload when reaching the minimum length of datasets. 

For more details please have a look at :paramref:`~pytorch_lightning.trainer.trainer.Trainer.Trainer.multiple_trainloader_mode`

----------

Multiple validation/test datasets
=================================
For validation and test DataLoaders, you can pass a single :class:`~torch.utils.data.DataLoader` or a list of them. This optional named
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


Test with additional dataloaders
=================================
You can run inference on a test set even if the :meth:`~pytorch_lightning.core.lightning.LightningModule.test_dataloader` method hasn't been
defined within your :class:`~pytorch_lightning.core.Lightning.LightningModule` instance. For example, rhis would be the case if your test data
set is not available at the time your model was declared. Simply pass the test set to the :meth:`~pytorch_lightning.trainer.trainer.Trainer.Trainer.test` method:

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

1. Return either a padded tensor in :class:`~torch.utils.data.Dataset` or a list of variable length tensors in the :class:`~torch.utils.data.DataLoader` collate_fn (example shows the list implementation).
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
    override :meth:`~pytorch_lightning.core.LightningModule.tbptt_split_batch`.

----------

Iterable Datasets
=================
Lightning supports using IterableDatasets as well as map-style Datasets. IterableDatasets provide a more natural
option when using sequential data.

.. note:: When using an IterableDataset you must set the ``val_check_interval`` to 1.0 (the default) or an int
    (specifying the number of training batches to run before validation) when initializing the Trainer. This is
    because the IterableDataset does not have a ``__len__`` and Lightning requires this to calculate the validation
    interval when ``val_check_interval`` is less than one. Similarly, you can set ``limit_{mode}_batches`` to a float or
    an int. If it is set to 0.0 or 0 it will set ``num_{mode}_batches`` to 0, if it is an int it will set ``num_{mode}_batches``
    to ``limit_{mode}_batches``, if it is set to 1.0 it will run for the whole Dataset, otherwise it will throw an exception.
    Here mode can be train/val/test.

.. testcode::

    # IterableDataSet
    class CustomDataset(IterableDataset):

        def __init__(self, data):
            self.data_source

        def __iter__(self):
            return iter(self.data_source)

    class LitModel(LightningModule):
        # Setup DataLoader
        def train_dataloader(self):
            seq_data = ['A', 'long', 'time', 'ago', 'in', 'a', 'galaxy', 'far', 'far', 'away']
            iterable_dataset = CustomDataset(seq_data)

            dataloader = DataLoader(Dataset=iterable_dataset, batch_size=5)
            return dataloader

.. testcode::

    # Set val_check_interval
    trainer = Trainer(val_check_interval=100)

    # Set limit_val_batches to 0.0 or 0
    trainer = Trainer(limit_val_batches=0.0)

    # Set limit_val_batches as an int
    trainer = Trainer(limit_val_batches=100)
