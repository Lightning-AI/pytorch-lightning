"""
A :class:`~LightningModule` organizes your PyTorch code into the following sections:

.. figure:: /_images/lightning_module/pt_to_pl.png
   :alt: Convert from PyTorch to Lightning


Notice a few things.

1.  It's the SAME code.
2.  The PyTorch code IS NOT abstracted - just organized.
3.  All the other code that's not in the :class:`~LightningModule`
    has been automated for you by the trainer.

    .. code-block:: python

        net = Net()
        trainer = Trainer()
        trainer.fit(net)

4.  There are no .cuda() or .to() calls... Lightning does these for you.

    .. code-block:: python

        # don't do in lightning
        x = torch.Tensor(2, 3)
        x = x.cuda()
        x = x.to(device)

        # do this instead
        x = x  # leave it alone!

        # or to init a new tensor
        new_x = torch.Tensor(2, 3)
        new_x = new_x.type_as(x.type())

5.  There are no samplers for distributed, Lightning also does this for you.

    .. code-block:: python

        # Don't do in Lightning...
        data = MNIST(...)
        sampler = DistributedSampler(data)
        DataLoader(data, sampler=sampler)

        # do this instead
        data = MNIST(...)
        DataLoader(data)

6.  A :class:`~LightningModule` is a :class:`torch.nn.Module` but with added functionality. Use it as such!

    .. code-block:: python

        net = Net.load_from_checkpoint(PATH)
        net.freeze()
        out = net(x)

Thus, to use Lightning, you just need to organize your code which takes about 30 minutes,
(and let's be real, you probably should do anyhow).

------------

Minimal Example
---------------

Here are the only required methods.

.. code-block:: python

    >>> import pytorch_lightning as pl
    >>> class LitModel(pl.LightningModule):
    ...
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.l1 = torch.nn.Linear(28 * 28, 10)
    ...
    ...     def forward(self, x):
    ...         return torch.relu(self.l1(x.view(x.size(0), -1)))
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         return {'loss': F.cross_entropy(y_hat, y)}
    ...
    ...     def train_dataloader(self):
    ...         return DataLoader(MNIST(os.getcwd(), train=True, download=True,
    ...                                 transform=transforms.ToTensor()), batch_size=32)
    ...
    ...     def configure_optimizers(self):
    ...         return torch.optim.Adam(self.parameters(), lr=0.02)

Which you can train by doing:

.. code-block:: python

   trainer = pl.Trainer()
   model = LitModel()

   trainer.fit(model)

----------

Training loop structure
-----------------------

The general pattern is that each loop (training, validation, test loop)
has 3 methods:

- ``___step``
- ``___step_end``
- ``___epoch_end``

To show how Lightning calls these, let's use the validation loop as an example:

.. code-block:: python

    val_outs = []
    for val_batch in val_data:
        # do something with each batch
        out = validation_step(val_batch)
        val_outs.append(out)

    # do something with the outputs for all batches
    # like calculate validation set accuracy or loss
    validation_epoch_end(val_outs)

If we use dp or ddp2 mode, we can also define the ``XXX_step_end`` method to operate
on all parts of the batch::

    val_outs = []
    for val_batch in val_data:
        batches = split_batch(val_batch)
        dp_outs = []
        for sub_batch in batches:
            dp_out = validation_step(sub_batch)
            dp_outs.append(dp_out)

        out = validation_step_end(dp_outs)
        val_outs.append(out)

    # do something with the outputs for all batches
    # like calculate validation set accuracy or loss
    validation_epoch_end(val_outs)


Add validation loop
^^^^^^^^^^^^^^^^^^^

Thus, if we wanted to add a validation loop you would add this to your
:class:`~LightningModule`:

    >>> class LitModel(pl.LightningModule):
    ...     def validation_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         return {'val_loss': F.cross_entropy(y_hat, y)}
    ...
    ...     def validation_epoch_end(self, outputs):
    ...         val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
    ...         return {'val_loss': val_loss_mean}
    ...
    ...     def val_dataloader(self):
    ...         # can also return a list of val dataloaders
    ...         return DataLoader(...)

Add test loop
^^^^^^^^^^^^^

    >>> class LitModel(pl.LightningModule):
    ...     def test_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         return {'test_loss': F.cross_entropy(y_hat, y)}
    ...
    ...     def test_epoch_end(self, outputs):
    ...         test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
    ...         return {'test_loss': test_loss_mean}
    ...
    ...     def test_dataloader(self):
    ...         # can also return a list of test dataloaders
    ...         return DataLoader(...)

However, the test loop won't ever be called automatically to make sure you
don't run your test data by accident. Instead you have to explicitly call:

.. code-block:: python

    # call after training
    trainer = Trainer()
    trainer.fit(model)
    trainer.test()

    # or call with pretrained model
    model = MyLightningModule.load_from_checkpoint(PATH)
    trainer = Trainer()
    trainer.test(model)

----------

Training_step_end method
------------------------
When using :class:`~pytorch_lightning.overrides.data_parallel.LightningDataParallel` or
:class:`~pytorch_lightning.overrides.data_parallel.LightningDistributedDataParallel`, the
:meth:`~LightningModule.training_step`
will be operating on a portion of the batch. This is normally ok but in special
cases like calculating NCE loss using negative samples, we might want to
perform a softmax across all samples in the batch.

For these types of situations, each loop has an additional ``__step_end`` method
which allows you to operate on the pieces of the batch:

.. code-block:: python

    training_outs = []
    for train_batch in train_data:
        # dp, ddp2 splits the batch
        sub_batches = split_batches_for_dp(batch)

        # run training_step on each piece of the batch
        batch_parts_outputs = [training_step(sub_batch) for sub_batch in sub_batches]

        # do softmax with all pieces
        out = training_step_end(batch_parts_outputs)
        training_outs.append(out)

    # do something with the outputs for all batches
    # like calculate validation set accuracy or loss
    training_epoch_end(val_outs)

----------

Remove cuda calls
-----------------
In a :class:`~LightningModule`, all calls to ``.cuda()``
and ``.to(device)`` should be removed. Lightning will do these
automatically. This will allow your code to work on CPUs, TPUs and GPUs.

When you init a new tensor in your code, just use :meth:`~torch.Tensor.type_as`:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        x, y = batch

        # put the z on the appropriate gpu or tpu core
        z = sample_noise()
        z = z.type_as(x)

----------

Data preparation
----------------
Data preparation in PyTorch follows 5 steps:

1. Download
2. Clean and (maybe) save to disk
3. Load inside :class:`~torch.utils.data.Dataset`
4. Apply transforms (rotate, tokenize, etc...)
5. Wrap inside a :class:`~torch.utils.data.DataLoader`

When working in distributed settings, steps 1 and 2 have to be done
from a single GPU, otherwise you will overwrite these files from
every GPU. The :class:`~LightningModule` has the
:class:`~LightningModule.prepare_data` method to
allow for this:

    >>> class LitModel(pl.LightningModule):
    ...     def prepare_data(self):
    ...         # download
    ...         mnist_train = MNIST(os.getcwd(), train=True, download=True,
    ...                             transform=transforms.ToTensor())
    ...         mnist_test = MNIST(os.getcwd(), train=False, download=True,
    ...                            transform=transforms.ToTensor())
    ...
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
    ...         return DataLoader(self.mnist_val, batch_size=64)
    ...
    ...     def test_dataloader(self):
    ...         return DataLoader(self.mnist_test, batch_size=64)

Note:
    :meth:`~LightningModule.prepare_data` is called once.

Note:
    Do anything with data that needs to happen ONLY once here, like download, tokenize, etc...


Lifecycle
---------
The methods in the :class:`~LightningModule` are called in this order:

1. :meth:`~LightningModule.__init__`
2. :meth:`~LightningModule.prepare_data`
3. :meth:`~LightningModule.configure_optimizers`
4. :meth:`~LightningModule.train_dataloader`

If you define a validation loop then

5. :meth:`~LightningModule.val_dataloader`

And if you define a test loop:

6. :meth:`~LightningModule.test_dataloader`

Note:
    :meth:`~LightningModule.test_dataloader` is only called with ``.test()``

In every epoch, the loop methods are called in this frequency:

1. :meth:`~LightningModule.validation_step` called every batch
2. :meth:`~LightningModule.validation_epoch_end` called every epoch

Live demo
---------
Check out this
`COLAB <https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg>`_
for a live demo.

LightningModule Class
---------------------

"""

from pytorch_lightning.core.decorators import data_loader
from pytorch_lightning.core.lightning import LightningModule

__all__ = ['LightningModule', 'data_loader']
# __call__ = __all__
