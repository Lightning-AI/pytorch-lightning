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
    ...         loss = F.cross_entropy(y_hat, y)
    ...         return pl.TrainResult(loss)
    ...
    ...     def configure_optimizers(self):
    ...         return torch.optim.Adam(self.parameters(), lr=0.02)

Which you can train by doing:

.. code-block:: python

    train_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()))
    trainer = pl.Trainer()
    model = LitModel()

   trainer.fit(model, train_loader)

----------

Training loop structure
-----------------------

The general pattern is that each loop has a single method to worry about

- ``___step``

If you need more control, there are two optional methods.

- ``___step_end``
- ``___epoch_end``

To show how Lightning calls these, let's use the validation loop as an example:

.. code-block:: python

    # put model in prediction mode
    model.eval()
    torch.set_grad_enabled(False)

    val_outs = []
    for val_batch in val_data:
        # do something with each batch
        out = validation_step(val_batch)
        val_outs.append(out)

    # do something with the outputs for all batches
    # like calculate validation set accuracy or loss
    validation_epoch_end(val_outs)

    # put model back in train mode
    model.train()
    torch.set_grad_enabled(True)

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

    >>> import pytorch_lightning as pl
    >>> class LitModel(pl.LightningModule):
    ...     def validation_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         loss = F.cross_entropy(y_hat, y)
    ...         result = pl.EvalResult(checkpoint_on=loss)
    ...         result.log('val_loss', loss)
    ...         return result

The equivalent expanded version (which you normally wouldn't need to use) is the following:

    >>> import pytorch_lightning as pl
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

    >>> import pytorch_lightning as pl
    >>> class LitModel(pl.LightningModule):
    ...     def test_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         loss = F.cross_entropy(y_hat, y)
    ...         result = pl.EvalResult(checkpoint_on=loss)
    ...         result.log('test_loss', loss)
    ...         return result

However, the test loop won't ever be called automatically to make sure you
don't run your test data by accident. Instead you have to explicitly call:

.. code-block:: python

    # call after training
    trainer = Trainer()
    trainer.fit(model)
    trainer.test(test_dataloaders=test_dataloader)

    # or call with pretrained model
    model = MyLightningModule.load_from_checkpoint(PATH)
    trainer = Trainer()
    trainer.test(model, test_dataloaders=test_dataloader)

-------------

Eval/TrainResult
^^^^^^^^^^^^^^^^
When you are using the `_step_end` and `_epoch_end` only for aggregating metrics and then logging,
consider using either a `EvalResult` or `TrainResult` instead.

Here's a traomg loop structure

.. code-block:: python

    def training_step(self, batch, batch_idx):
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'epoch_loss': epoch_loss},
            'progress_bar': {'epoch_loss': epoch_loss}
        }

using the equivalent syntax via the `TrainResult` object:

.. code-block:: python

    def training_step(self, batch_subset, batch_idx):
        loss = ...
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

Same for val/test loop

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        return {'some_metric': some_metric}

    def validation_epoch_end(self, validation_step_outputs):
        some_metric_mean = torch.stack([x['some_metric'] for x in validation_step_outputs]).mean()
        return {
            'log': {'some_metric_mean': some_metric_mean},
            'progress_bar': {'some_metric_mean': some_metric_mean}
        }

With the equivalent using the `EvalResult` syntax

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        some_metric = ...
        result = pl.EvalResult(checkpoint_on=some_metric)
        result.log('some_metric', some_metric, prog_bar=True)
        return result

----------

Training_step_end method
------------------------
When using :class:`~pytorch_lightning.overrides.data_parallel.LightningDataParallel` or
:class:`~pytorch_lightning.overrides.data_parallel.LightningDistributedDataParallel`, the
:meth:`~LightningModule.training_step`
will be operating on a portion of the batch. This is normally okay but in special
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

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule

__all__ = [
    'LightningDataModule',
    'LightningModule',
]
# __call__ = __all__
