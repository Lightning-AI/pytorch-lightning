Training loop
===============

Accumulate gradients
-------------------------------------
Accumulated gradients runs K small batches of size N before doing a backwards pass.
The effect is a large effective batch size of size KxN.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

Packed sequences as inputs
----------------------------
When using PackedSequence, do 2 things:

1. return either a padded tensor in dataset or a list of variable length tensors in the dataloader collate_fn (example above shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. code-block:: python

   # For use in dataloader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y

    # In module
    def training_step(self, batch, batch_nb):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)


Truncated Backpropagation Through Time
---------------------------------------------
There are times when multiple backwards passes are needed for each batch.
For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

Lightning can handle TBTT automatically via this flag.

.. code-block:: python

    # DEFAULT (single backwards pass per batch)
    trainer = Trainer(truncated_bptt_steps=None)

    # (split batch into sequences of size 2)
    trainer = Trainer(truncated_bptt_steps=2)

.. note:: If you need to modify how the batch is split,
    override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`.

.. note:: Using this feature requires updating your LightningModule's :meth:`pytorch_lightning.core.LightningModule.training_step` to include
    a `hiddens` arg.


Force training for min or max epochs
-------------------------------------
It can be useful to force training for a minimum number of epochs or limit to a max number.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)

Early stopping callback
-------------------------------------
There are two ways to enable early stopping.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # A) Looks for val_loss in validation_step return dict
    trainer = Trainer(early_stop_callback=True)

    # B) Or configure your own callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(early_stop_callback=early_stop_callback)

Force disable early stop
-------------------------------------
To disable early stopping pass None to the early_stop_callback

.. note:: See: :ref:`trainer`

.. code-block:: python

   # DEFAULT
   trainer = Trainer(early_stop_callback=None)


Gradient Clipping
-------------------------------------
Gradient clipping may be enabled to avoid exploding gradients. Specifically, this will `clip the gradient
norm <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_ computed over all model parameters together.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)


Learning rate scheduling
-------------------------------------
Every optimizer you use can be paired with any `LearningRateScheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

.. code-block:: python

   # no LR scheduler
   def configure_optimizers(self):
      return Adam(...)

   # Adam + LR scheduler
   def configure_optimizers(self):
      return [Adam(...)], [ReduceLROnPlateau()]

   # Two optimziers each with a scheduler
   def configure_optimizers(self):
      return [Adam(...), SGD(...)], [ReduceLROnPlateau(), LambdaLR()]


Use multiple optimizers (like GANs)
-------------------------------------
To use multiple optimizers return > 1 optimizers from :meth:`pytorch_lightning.core.LightningModule.configure_optimizers`

.. code-block:: python

   # one optimizer
   def configure_optimizers(self):
      return Adam(...)

   # two optimizers, no schedulers
   def configure_optimizers(self):
      return Adam(...), SGD(...)

   # Two optimizers, one scheduler for adam only
   def configure_optimizers(self):
      return [Adam(...), SGD(...)], [ReduceLROnPlateau()]

Lightning will call each optimizer sequentially:

.. code-block:: python

   for epoch in epochs:
      for batch in data:
         for opt in optimizers:
            train_step(opt)
            opt.step()

      for scheduler in scheduler:
         scheduler.step()

Set how much of the training set to check (1-100%)
---------------------------------------------------
If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

.. code-block:: python

   # DEFAULT
   trainer = Trainer(train_percent_check=1.0)

   # check 10% only
   trainer = Trainer(train_percent_check=0.1)

.. note:: train_percent_check will be overwritten by overfit_pct if overfit_pct > 0

Step optimizers at arbitrary intervals
-------------------------------------
To do more interesting things with your optimizers such as learning rate warm-up or odd scheduling,
override the :meth:`optimizer_step' function.

For example, here step optimizer A every 2 batches and optimizer B every 4 batches

.. code-block:: python

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()

    # Alternating schedule for optimizer steps (ie: GANs)
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # update generator opt every 2 steps
        if optimizer_i == 0:
            if batch_nb % 2 == 0 :
                optimizer.step()
                optimizer.zero_grad()

        # update discriminator opt every 4 steps
        if optimizer_i == 1:
            if batch_nb % 4 == 0 :
                optimizer.step()
                optimizer.zero_grad()

        # ...
        # add as many optimizers as you want

Here we add a learning-rate warm up

.. code-block:: python

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()

Validation loop
================

Check validation every n epochs
-------------------------------------
If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)


Set how much of the validation set to check
--------------------------------------------
If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag
val_percent_check will be overwritten by overfit_pct if overfit_pct > 0

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_percent_check=1.0)

    # check 10% only
    trainer = Trainer(val_percent_check=0.1)


Set how much of the test set to check
-------------------------------------
If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag
test_percent_check will be overwritten by overfit_pct if overfit_pct > 0.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(test_percent_check=1.0)

    # check 10% only
    trainer = Trainer(test_percent_check=0.1)


Set validation check frequency within 1 training epoch
-------------------------------------
For large datasets it's often desirable to check validation multiple times within a training loop.
Pass in a float to check that often within 1 training epoch. Pass in an int k to check every k training batches.
Must use an int if using an IterableDataset.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)

Set the number of validation sanity steps
-------------------------------------
Lightning runs a few steps of validation in the beginning of training.
This avoids crashing in the validation loop sometime deep into a lengthy training loop.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(nb_sanity_val_steps=5)

Testing loop
=============

Run test set
-------------------------------------
You have two options to run the test set. First case is where you test right after a full training routine.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # run test set
    trainer.test()

Second case is where you load a model and run the test set

.. code-block:: python

    model = MyLightningModule.load_from_metrics(
        weights_path='/path/to/pytorch_checkpoint.ckpt',
        tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
        on_gpu=True,
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this second case, the options you pass to trainer will be used when
running the test set (ie: 16-bit, dp, ddp, etc...)

Examples
=============================

.. toctree::
   :maxdepth: 3

   pl_examples

