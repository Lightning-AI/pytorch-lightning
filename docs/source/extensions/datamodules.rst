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
    trainer.fit(model, datamodule=imagenet)

    cifar10 = CIFAR10DataModule()
    trainer.fit(model, datamodule=cifar10)

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
A DataModule is simply a collection of a train_dataloader(s), val_dataloader(s), test_dataloader(s) and
predict_dataloader(s) along with the matching transforms and data processing/downloads steps required.

Here's a simple PyTorch example:

.. code-block:: python

    # regular PyTorch
    test_data = MNIST(my_path, train=False, download=True)
    predict_data = MNIST(my_path, train=False, download=True)
    train_data = MNIST(my_path, train=True, download=True)
    train_data, val_data = random_split(train_data, [55000, 5000])

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    predict_loader = DataLoader(predict_data, batch_size=32)

The equivalent DataModule just organizes the same exact code, but makes it reusable across projects.

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size

        def setup(self, stage: Optional[str] = None):
            self.mnist_test = MNIST(self.data_dir, train=False)
            self.mnist_predict = MNIST(self.data_dir, train=False)
            mnist_full = MNIST(self.data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=self.batch_size)

        def predict_dataloader(self):
            return DataLoader(self.mnist_predict, batch_size=self.batch_size)

        def teardown(self, stage: Optional[str] = None):
            # Used to clean-up when the run is finished
            ...

But now, as the complexity of your processing grows (transforms, multiple-GPU training), you can
let Lightning handle those details for you while making this dataset reusable so you can share with
colleagues or use in different projects.

.. code-block:: python

    mnist = MNISTDataModule(my_path)
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
        def __init__(self, data_dir: str = "./"):
            super().__init__()
            self.data_dir = data_dir
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        def prepare_data(self):
            # download
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

        def setup(self, stage: Optional[str] = None):

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            if stage == "predict" or stage is None:
                self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=32)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=32)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=32)

        def predict_dataloader(self):
            return DataLoader(self.mnist_predict, batch_size=32)

---------------

LightningDataModule API
-----------------------
To define a DataModule the following methods are used to create train/val/test/predict dataloaders:

- :ref:`prepare_data<extensions/datamodules:prepare_data>` (how to download, tokenize, etc...)
- :ref:`setup<extensions/datamodules:setup>` (how to split, define dataset, etc...)
- :ref:`train_dataloader<extensions/datamodules:train_dataloader>`
- :ref:`val_dataloader<extensions/datamodules:val_dataloader>`
- :ref:`test_dataloader<extensions/datamodules:test_dataloader>`
- :ref:`predict_dataloader<extensions/datamodules:predict_dataloader>`


prepare_data
~~~~~~~~~~~~
Downloading and saving data with multiple processes (distributed settings) will result in corrupted data. Lightning
ensures the :meth:`~pytorch_lightning.core.hooks.DataHooks.prepare_data` is called only within a single process,
so you can safely add your downloading logic within. In case of multi-node training, the execution of this hook
depends upon :ref:`prepare_data_per_node<extensions/datamodules:prepare_data_per_node>`.

- download
- tokenize
- etc...

.. code-block:: python

    class MNISTDataModule(pl.LightningDataModule):
        def prepare_data(self):
            # download
            MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
            MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())


.. warning:: ``prepare_data`` is called from the main process. It is not recommended to assign state here (e.g. ``self.x = y``).


setup
~~~~~
There are also data operations you might want to perform on every GPU. Use :meth:`~pytorch_lightning.core.hooks.DataHooks.setup` to do things like:

- count number of classes
- build vocabulary
- perform train/val/test splits
- create datasets
- apply transforms (defined explicitly in your datamodule)
- etc...

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def setup(self, stage: Optional[str] = None):

            # Assign Train/val split(s) for use in Dataloaders
            if stage in (None, "fit"):
                mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Assign Test split(s) for use in Dataloaders
            if stage in (None, "test"):
                self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)


This method expects a ``stage`` argument.
It is used to separate setup logic for ``trainer.{fit,validate,test,predict}``. If ``setup`` is called with ``stage=None``,
we assume all stages have been set-up.

.. note:: :ref:`setup<extensions/datamodules:setup>` is called from every process across all the nodes. Setting state here is recommended.
.. note:: :ref:`teardown<extensions/datamodules:teardown>` can be used to clean up the state. It is also called from every process across all the nodes.


train_dataloader
~~~~~~~~~~~~~~~~
Use the :meth:`~pytorch_lightning.core.hooks.DataHooks.train_dataloader` method to generate the training dataloader(s).
Usually you just wrap the dataset you defined in :ref:`setup<extensions/datamodules:setup>`. This is the dataloader that the Trainer
:meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` method uses.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=64)

.. _datamodule_val_dataloader_label:

val_dataloader
~~~~~~~~~~~~~~
Use the :meth:`~pytorch_lightning.core.hooks.DataHooks.val_dataloader` method to generate the validation dataloader(s).
Usually you just wrap the dataset you defined in :ref:`setup<extensions/datamodules:setup>`. This is the dataloader that the Trainer
:meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` and :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate` methods uses.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=64)


.. _datamodule_test_dataloader_label:

test_dataloader
~~~~~~~~~~~~~~~
Use the :meth:`~pytorch_lightning.core.hooks.DataHooks.test_dataloader` method to generate the test dataloader(s).
Usually you just wrap the dataset you defined in :ref:`setup<extensions/datamodules:setup>`. This is the dataloader that the Trainer
:meth:`~pytorch_lightning.trainer.trainer.Trainer.test` method uses.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=64)


predict_dataloader
~~~~~~~~~~~~~~~~~~
Use the :meth:`~pytorch_lightning.core.hooks.DataHooks.predict_dataloader` method to generate the prediction dataloader(s).
Usually you just wrap the dataset you defined in :ref:`setup<extensions/datamodules:setup>`. This is the dataloader that the Trainer
:meth:`~pytorch_lightning.trainer.trainer.Trainer.predict` method uses.

.. code-block:: python

    import pytorch_lightning as pl


    class MNISTDataModule(pl.LightningDataModule):
        def predict_dataloader(self):
            return DataLoader(self.mnist_predict, batch_size=64)


transfer_batch_to_device
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.transfer_batch_to_device
    :noindex:

on_before_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_before_batch_transfer
    :noindex:

on_after_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_after_batch_transfer
    :noindex:

load_state_dict
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.load_state_dict
    :noindex:

state_dict
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.state_dict
    :noindex:

on_train_dataloader
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_train_dataloader
    :noindex:

on_val_dataloader
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_val_dataloader
    :noindex:

on_test_dataloader
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_test_dataloader
    :noindex:

on_predict_dataloader
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.on_predict_dataloader
    :noindex:

teardown
~~~~~~~~

.. automethod:: pytorch_lightning.core.datamodule.LightningDataModule.teardown
    :noindex:

prepare_data_per_node
~~~~~~~~~~~~~~~~~~~~~
If set to ``True`` will call ``prepare_data()`` on LOCAL_RANK=0 for every node.
If set to ``False`` will only call from NODE_RANK=0, LOCAL_RANK=0.

.. testcode::

    class LitDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()
            self.prepare_data_per_node = True


------------------

Using a DataModule
------------------

The recommended way to use a DataModule is simply:

.. code-block:: python

    dm = MNISTDataModule()
    model = Model()
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.predict(datamodule=dm)

If you need information from the dataset to build your model, then run
:ref:`prepare_data<extensions/datamodules:prepare_data>` and
:ref:`setup<extensions/datamodules:setup>` manually (Lightning ensures
the method runs on the correct devices).

.. code-block:: python

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")

    model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)
    trainer.fit(model, dm)

    dm.setup(stage="test")
    trainer.test(datamodule=dm)

----------------

DataModules without Lightning
-----------------------------
You can of course use DataModules in plain PyTorch code as well.

.. code-block:: python

    # download, etc...
    dm = MNISTDataModule()
    dm.prepare_data()

    # splits/transforms
    dm.setup(stage="fit")

    # use data
    for batch in dm.train_dataloader():
        ...

    for batch in dm.val_dataloader():
        ...

    dm.teardown(stage="fit")

    # lazy load test data
    dm.setup(stage="test")
    for batch in dm.test_dataloader():
        ...

    dm.teardown(stage="test")

But overall, DataModules encourage reproducibility by allowing all details of a dataset to be specified in a unified
structure.

----------------

Hyperparameters in DataModules
------------------------------
Like LightningModules, DataModules support hyperparameters with the same API.

.. code-block:: python

    import pytorch_lightning as pl


    class CustomDataModule(pl.LightningDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

        def configure_optimizers(self):
            # access the saved hyperparameters
            opt = optim.Adam(self.parameters(), lr=self.hparams.lr)

Refer to ``save_hyperparameters`` in :doc:`lightning module <../common/lightning_module>` for more details.
