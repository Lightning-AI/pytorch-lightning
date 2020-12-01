.. _datamodules:

LightningDataModule
===================
A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data:

.. raw:: html

    <video width="100%" max-width="400px" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_dm_vid.m4v"></video>

|

A datamodule encapsulates the five steps involved in data processing in PyTorch:

1. Download / tokenize / process.
2. Clean and (maybe) save to disk.
3. Load inside :class:`~torch.utils.data.Dataset`.
4. Apply transforms (rotate, tokenize, etc...).
5. Wrap inside a :class:`~torch.utils.data.DataLoader`.

|

This class can then be shared and used anywhere:

.. code-block:: python

    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

    model = LitClassifier()
    trainer = Trainer()

    imagenet = ImagenetDataModule()
    trainer.fit(model, imagenet)

    cifar10 = CIFAR10DataModule()
    trainer.fit(model, cifar10)

---------------

Why do I need a DataModule?
---------------------------
In normal PyTorch code, the data cleaning/preparation is usually scattered across many files. This makes
sharing and reusing the exact splits and transforms across projects impossible.

Datamodules are for you if you ever asked the questions:

- what splits did you use?
- what transforms did you use?
- what normalization did you use?
- how did you prepare/tokenize the data?

--------------

What is a DataModule
--------------------
A DataModule is simply a collection of a train_dataloader, val_dataloader(s), test_dataloader(s) along with the
matching transforms and data processing/downloads steps required.

Here's a simple PyTorch example:

.. code-block:: python

    # regular PyTorch
    test_data = MNIST(PATH, train=False, download=True)
    train_data = MNIST(PATH, train=True, download=True)
    train_data, val_data = random_split(train_data, [55000, 5000])

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

The equivalent DataModule just organizes the same exact code, but makes it reusable across projects.

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):

        def __init__(self, data_dir: str = PATH, batch_size):
            super().__init__()
            self.batch_size = batch_size

        def setup(self, stage=None):
            self.mnist_test = MNIST(self.data_dir, train=False)
            mnist_full = MNIST(self.data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=self.batch_size)

But now, as the complexity of your processing grows (transforms, multiple-GPU training), you can
let Lightning handle those details for you while making this dataset reusable so you can share with
colleagues or use in different projects.

.. code-block:: python

    mnist = MNISTDataModule(PATH)
    model = LitClassifier()

    trainer = Trainer()
    trainer.fit(model, mnist)

Here's a more realistic, complex DataModule that shows how much more reusable the datamodule is.

.. code-block:: python

    import pytorch_lightning as pl
    from torch.utils.data import random_split, DataLoader

    # Note - you must have torchvision installed for this example
    from torchvision.datasets import MNIST
    from torchvision import transforms


    class MNISTDataModule(pl.LightningDataModule):

        def __init__(self, data_dir: str = './'):
            super().__init__()
            self.data_dir = data_dir
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # self.dims is returned when you call dm.size()
            # Setting default dims here because we know them.
            # Could optionally be assigned dynamically in dm.setup()
            self.dims = (1, 28, 28)

        def prepare_data(self):
            # download
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

        def setup(self, stage=None):

            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

                # Optionally...
                # self.dims = tuple(self.mnist_train[0][0].shape)

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

                # Optionally...
                # self.dims = tuple(self.mnist_test[0][0].shape)

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=32)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=32)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=32)

.. note:: ``setup`` expects a string arg ``stage``. It is used to separate setup logic for ``trainer.fit`` and ``trainer.test``.


---------------

LightningDataModule API
-----------------------
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

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):
        def prepare_data(self):
            # download
            MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
            MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

.. warning:: `prepare_data` is called from a single GPU. Do not use it to assign state (`self.x = y`).

setup
^^^^^
There are also data operations you might want to perform on every GPU. Use setup to do things like:

- count number of classes
- build vocabulary
- perform train/val/test splits
- apply transforms (defined explicitly in your datamodule or assigned in init)
- etc...

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):

        def setup(self, stage: Optional[str] = None):

            # Assign Train/val split(s) for use in Dataloaders
            if stage == 'fit' or stage is None:
                mnist_full = MNIST(
                    self.data_dir,
                    train=True,
                    download=True,
                    transform=self.transform
                )
                self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
                self.dims = self.mnist_train[0][0].shape

            # Assign Test split(s) for use in Dataloaders
            if stage == 'test' or stage is None:
                self.mnist_test = MNIST(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self.transform
                )
                self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)


.. warning:: `setup` is called from every GPU. Setting state here is okay.


train_dataloader
^^^^^^^^^^^^^^^^
Use this method to generate the train dataloader.  Usually you just wrap the dataset you defined in ``setup``.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=64)


val_dataloader
^^^^^^^^^^^^^^
Use this method to generate the val dataloader.  Usually you just wrap the dataset you defined in ``setup``.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=64)

.. _datamodule-test-dataloader-label:

test_dataloader
^^^^^^^^^^^^^^^
Use this method to generate the test dataloader. Usually you just wrap the dataset you defined in ``setup``.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=64)

transfer_batch_to_device
^^^^^^^^^^^^^^^^^^^^^^^^
Override to define how you want to move an arbitrary batch to a device

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def transfer_batch_to_device(self, batch, device):
            x = batch['x']
            x = CustomDataWrapper(x)
            batch['x'].to(device)
            return batch


.. note:: To decouple your data from transforms you can parametrize them via `__init__`.

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, train_transforms, val_transforms, test_transforms):
            super().__init__()
            self.train_transforms = train_transforms
            self.val_transforms = val_transforms
            self.test_transforms = test_transforms


------------------

Using a DataModule
------------------

The recommended way to use a DataModule is simply:

.. code-block:: python

    dm = MNISTDataModule()
    model = Model()
    trainer.fit(model, dm)

    trainer.test(datamodule=dm)

If you need information from the dataset to build your model, then run `prepare_data` and `setup` manually (Lightning
still ensures the method runs on the correct devices)

.. code-block:: python

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup('fit')

    model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)
    trainer.fit(model, dm)

    dm.setup('test')
    trainer.test(datamodule=dm)

----------------

Datamodules without Lightning
-----------------------------
You can of course use DataModules in plain PyTorch code as well.

.. code-block:: python

    # download, etc...
    dm = MNISTDataModule()
    dm.prepare_data()

    # splits/transforms
    dm.setup('fit')

    # use data
    for batch in dm.train_dataloader():
        ...
    for batch in dm.val_dataloader():
        ...

    # lazy load test data
    dm.setup('test')
    for batch in dm.test_dataloader():
        ...

But overall, DataModules encourage reproducibility by allowing all details of a dataset to be specified in a unified
structure.
