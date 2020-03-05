"""
A LightningModule organizes your PyTorch code into the following sections:

- The model/system definition (__init__)
- The model/system computations (forward)
- What happens in the training loop (training_step)
- What happens in the validation loop (validation_step, validation_epoch_end)
- What happens in the test loop (test_step, test_epoch_end)
- What optimizers to use (configure_optimizers)
- What data to use (train_dataloader, val_dataloader, test_dataloader)

.. note:: LightningModule is a torch.nn.Module but with added functionality.

------------

Minimal Example
---------------

Most methods are optional. Here's a minimal example.

.. code-block:: python

    import os
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    import pytorch_lightning as pl

    class CoolModel(pl.LightningModule):

        def __init__(self):
            super(CoolModel, self).__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)
            return {'loss': F.cross_entropy(y_hat, y)}

        def train_dataloader(self):
            return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                              transform=transforms.ToTensor()), batch_size=32)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

Which you can train by doing:

.. code-block:: python

   trainer = pl.Trainer()
   model = CoolModel()

   trainer.fit(model)

----------

Training loop structure
-----------------------

The general pattern is that each loop (training, validation, test loop)
has 2 methods:

- ``` ___step ```
- ``` ___epoch_end```

To show how lightning calls these, let's use the validation loop as an example

.. code-block:: python

    val_outs = []
    for val_batch in val_data:
        # do something with each batch
        out = validation_step(val_batch)
        val_outs.append(out)

    # do something with the outputs for all batches
    # like calculate validation set accuracy or loss
    validation_epoch_end(val_outs)

Add validation loop
^^^^^^^^^^^^^^^^^^^

Thus, if we wanted to add a validation loop you would add this to your LightningModule

.. code-block:: python

        class CoolModel(pl.LightningModule):
            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                return {'val_loss': F.cross_entropy(y_hat, y)}

            def validation_epoch_end(self, outputs):
                val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
                return {'val_loss': val_loss_mean}

            def val_dataloader(self):
                # can also return a list of val dataloaders
                return DataLoader(...)

Add test loop
^^^^^^^^^^^^^

.. code-block:: python

        class CoolModel(pl.LightningModule):
            def test_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                return {'test_loss': F.cross_entropy(y_hat, y)}

            def test_epoch_end(self, outputs):
                test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
                return {'test_loss': test_loss_mean}

            def test_dataloader(self):
                # can also return a list of test dataloaders
                return DataLoader(...)

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

Training_step_end method
------------------------
When using dataParallel or distributedDataParallel2, the training_step
will be operating on a portion of the batch. This is normally ok but in special
cases like calculating NCE loss using negative samples, we might want to
perform a softmax across all samples in the batch.

For these types of situations, each loop has an additional ```__step_end``` method
which allows you to operate on the pieces of the batch

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

.cuda, .to
----------
In a LightningModule, all calls to .cuda and .to should be removed. Lightning will do these
automatically. This will allow your code to work on CPUs, TPUs and GPUs.

When you init a new tensor in your code, just use type_as

.. code-block:: python

    def training_step(self, batch, batch_idx):
        x, y = batch

        # put the z on the appropriate gpu or tpu core
        z = sample_noise()
        z = z.type_as(x.type())

Live demo
---------
Check out how this live demo
Check out this
`COLAB <https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg>`_
for a live demo.

.. note:: Remove all .cuda() or .to() calls from LightningModules. See:
    `the multi-gpu training guide for details <multi_gpu.rst>`_.

"""

from .decorators import data_loader
from .lightning import LightningModule

__all__ = ['LightningModule', 'data_loader']
# __call__ = __all__
