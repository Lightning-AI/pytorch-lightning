LightningDataModule
===================
Data preparation in PyTorch follows 5 steps:

1. Download / tokenize / process.
2. Clean and (maybe) save to disk.
3. Load inside :class:`~torch.utils.data.Dataset`.
4. Apply transforms (rotate, tokenize, etc...).
5. Wrap inside a :class:`~torch.utils.data.DataLoader`.

A DataModule is simply a collection of a train_dataloader, val_dataloader(s), test_dataloader(s) along with the
matching transforms and data processing/downloads steps required.


    >>> import pytorch_lightning as pl
    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def prepare_data(self):
    ...         # download
    ...         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    ...         MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    ...
    ...     def setup(self, stage):
    ...         mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
    ...         mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
    ...         # train/val split
    ...         mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
    ...
    ...         # assign to use in dataloaders
    ...         self.train_dataset = mnist_train
    ...         self.val_dataset = mnist_val
    ...         self.test_dataset = mnist_test
    ...
    ...     def train_dataloader(self):
    ...         return DataLoader(self.train_dataset, batch_size=64)
    ...
    ...     def val_dataloader(self):
    ...         return DataLoader(self.val_dataset, batch_size=64)
    ...
    ...     def test_dataloader(self):
    ...         return DataLoader(self.test_dataset, batch_size=64)

---------------

Methods
-------
To define a DataModule define 5 methods:

- prepare_data (how to download(), tokenize, etc...)
- setup (how to split, etc...)
- train_dataloader
- val_dataloader(s)
- test_dataloader(s)

prepare_data
^^^^^^^^^^^^
Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed
settings.

- download
- tokenize
- etc...

    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def prepare_data(self):
    ...         # download
    ...         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    ...         MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

.. warning:: `prepare_data` is called from a single GPU. Do not use it to assign state (`self.x = y`).

setup
^^^^^
There are also data operations you might want to perform on every GPU. Use setup to do things like:

- count number of classes
- build vocabulary
- perform train/val/test splits
- etc...

    >>> import pytorch_lightning as pl
    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def setup(self, stage):
    ...         mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
    ...         mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
    ...         # train/val split
    ...         mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
    ...
    ...         # assign to use in dataloaders
    ...         self.train_dataset = mnist_train
    ...         self.val_dataset = mnist_val
    ...         self.test_dataset = mnist_test

.. warning:: `setup` is called from every GPU. Setting state here is okay.

train_dataloader
^^^^^^^^^^^^^^^^
Use this method to generate the train dataloader. This is also a good place to place default transformations.

    >>> import pytorch_lightning as pl
    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def train_dataloader(self):
    ...         transforms = transform_lib.Compose([
    ...             transform_lib.ToTensor(),
    ...             transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
    ...         ])
    ...         return DataLoader(self.train_dataset, transform=transforms, batch_size=64)

However, to decouple your data from transforms you can parametrize them via `__init__`.

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, train_transforms, val_transforms, test_transforms):
            self.train_transforms = train_transforms
            self.val_transforms = val_transforms
            self.test_transforms = test_transforms

val_dataloader
^^^^^^^^^^^^^^
Use this method to generate the val dataloader. This is also a good place to place default transformations.

    >>> import pytorch_lightning as pl
    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def val_dataloader(self):
    ...         transforms = transform_lib.Compose([
    ...             transform_lib.ToTensor(),
    ...             transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
    ...         ])
    ...         return DataLoader(self.val_dataset, transform=transforms, batch_size=64)

test_dataloader
^^^^^^^^^^^^^^^
Use this method to generate the test dataloader. This is also a good place to place default transformations.

    >>> import pytorch_lightning as pl
    >>> class MNISTDataModule(pl.LightningDataModule):
    ...     def test_dataloader(self):
    ...         transforms = transform_lib.Compose([
    ...             transform_lib.ToTensor(),
    ...             transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
    ...         ])
    ...         return DataLoader(self.test_dataset, transform=transforms, batch_size=64)

------------------

Using a DataModule
------------------
The recommended way to use a DataModule is simply:

.. code-block:: python

    dm = MNISTDataModule()
    model = Model()
    trainer.fit(model, datamodule=dm)

    trainer.test(datamodule=dm)

If you need information from the dataset to build your model, then run `prepare_data` and `setup` manually (Lightning
still ensures the method runs on the correct devices)

.. code-block:: python

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

----------------

Why use datamodules?
--------------------
DataModules have a few key advantages:

- It decouples the data from the model.
- It has all the necessary details for anyone to use the exact same data setup.
- Datamodules can be shared across models.
- Datamodules can also be used without Lightning by calling the methods directly

.. code-block:: python

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():
        ...
    for batch in dm.val_dataloader():
        ...
    for batch in dm.test_dataloader():
        ...

But overall, DataModules encourage reproducibility by allowing all details of a dataset to be specified in a unified
structure.
