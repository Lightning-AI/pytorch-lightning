Debugging
==========
The following are flags that make debugging much easier.

Fast dev run
-------------------
This flag runs a "unit test" by running 1 training batch and 1 validation batch.
The point is to detect any bugs in the training/validation loop without having to wait for
a full epoch to crash.

.. code-block:: python

    trainer = pl.Trainer(fast_dev_run=True)

Inspect gradient norms
-----------------------------------
Logs (to a logger), the norm of each weight matrix.

.. code-block:: python

    # the 2-norm
    trainer = pl.Trainer(track_grad_norm=2)

Log GPU usage
-----------------------------------
Logs (to a logger) the GPU usage for each GPU on the master machine.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(log_gpu_memory=True)

Make model overfit on subset of data
-----------------------------------

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(overfit_pct=0.01)

Print the parameter count by layer
-----------------------------------
Whenever the .fit() function gets called, the Trainer will print the weights summary for the lightningModule.
To disable this behavior, turn off this flag:

(See: :ref:`trainer.weights_summary`)

.. code-block:: python

    trainer = pl.Trainer(weights_summary=None)

Print which gradients are nan
------------------------------
Prints the tensors with nan gradients.

(See: :meth:`trainer.print_nan_grads`)

.. code-block:: python

    trainer = pl.Trainer(print_nan_grads=False)

Distributed training
=====================

Implement Your Own Distributed (DDP) training
----------------------------------------------
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.core.LightningModule.init_ddp_connection`.

If you also need to use your own DDP implementation, override:  :meth:`pytorch_lightning.core.LightningModule.configure_ddp`.

16-bit mixed precision
----------------------------------------------
To use 16-bit precision, do two things:

1. Install apex

.. code-block:: bash

    $ git clone https://github.com/NVIDIA/apex
    $ cd apex

    # ------------------------
    # OPTIONAL: on your cluster you might need to load cuda 10 or 9
    # depending on how you installed PyTorch

    # see available modules
    module avail

    # load correct cuda before install
    module load cuda-10.0
    # ------------------------

    # make sure you've loaded a cuda version > 4.0 and < 7.0
    module load gcc-6.1.0

    $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


2. Set the trainer flag

.. code-block:: python

    # DEFAULT
    trainer = Trainer(amp_level='O1', use_amp=False)

If you need to configure the apex init for your particular use case or want to use a different way of doing
16-bit training, override   :meth:`pytorch_lightning.core.LightningModule.configure_apex`.

Multi-GPU
----------------------------------------------
To train on multiple GPUs make sure you are running lightning on a machine with GPUs. Lightning handles
all the NVIDIA flags for you, there's no need to set them yourself.

There are three options for multi-GPU training:

1. DataParallel (dp) - Splits a batch across GPUs on a single machine.

2. DistributedDataParallel (ddp) - Splits data across each GPU and only syncs gradients.

3. ddp2 - Acts like dp on a single machine but syncs gradients across machines like ddp.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=1, distributed_backend='dp')

    # train on 2 GPUs (using dp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='dp')

    # train on 2 GPUs (using ddp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='ddp')

    # train on 0 GPUs
    trainer = pl.Trainer()


Multi-node
----------------------------------------------
See :ref:`multi-node`

Single GPU
----------------------------------------------
Make sure you are running on a machine that has at least one GPU. Lightning handles all the NVIDIA flags for you,
there's no need to set them yourself.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=1)


Experiment Logging
====================
Lightning supports many different experiment loggers. These loggers allow you to monitor losses, images, text, etc...
as training progresses. They usually provide a GUI to visualize and can sometimes even snapshot hyperparameters
used in each experiment.

Log/plot any metrics
----------------------------------------------
To plot metrics into whatever logger you passed in (tensorboard, comet, neptune, etc...)

1. Training_end, validation_end, test_end will all log anything in the "log" key of the return dict.

.. code-block:: python

   def training_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'train_loss': loss}
      results = {'log': logs}
      return results

   def validation_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'val_loss': loss}
      results = {'log': logs}
      return results

   def test_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'test_loss': loss}
      results = {'log': logs}
      return results

2. Most of the time, you only need training_step and not training_end. You can also return logs from here:

.. code-block:: python

   def training_step(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'train_loss': loss}
      results = {'log': logs}
      return results

3. In addition, you can also use any arbitrary functionality from a particular logger from within your LightningModule.
For instance, here we log images using tensorboard.

.. code-block:: python

   def training_step(self, batch, batch_idx):
      self.generated_imgs = self.decoder.generate()

      sample_imgs = self.generated_imgs[:6]
      grid = torchvision.utils.make_grid(sample_imgs)
      self.logger.experiment.add_image('generated_images', grid, 0)

      ...
      return results

Display metrics in the progress bar
----------------------------------------------
Each return dict from the training_end, validation_end, testing_end and training_step also has
a key called "progress_bar".

Here we show the validation loss in the progress bar

.. code-block:: python

   def validation_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'val_loss': loss}
      results = {'progress_bar': logs}
      return results


Log metric row every k batches
----------------------------------------------
It may slow training down to log every single batch. Trainer has an option to log every k batches instead.

.. code-block:: python

   # k = 10
   Trainer(row_log_interval=10)


Tensorboard
----------------------------------------------
To use `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ as your logger do the following.

.. note:: See: TensorBoardLogger :ref:`tf-logger`

.. code-block:: python

   from pytorch_lightning.loggers import TensorBoardLogger

   logger = TensorBoardLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TensorBoardLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)


Test Tube
----------------------------------------------
`Test Tube <https://github.com/williamFalcon/test-tube>`_ is a tensorboard logger but with nicer file structure.
To use TestTube as your logger do the following.

.. note:: See: TestTube :ref:`testTube`

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

   logger = TestTubeLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TestTubeLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Comet.ml
----------------------------------------------
`Comet.ml <https://www.comet.ml/site/>`_ is a third-party logger.
To use CometLogger as your logger do the following.

.. note:: See: :ref:`comet` docs.

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        workspace=os.environ["COMET_WORKSPACE"], # Optional
        project_name="default_project", # Optional
        rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )
   trainer = Trainer(logger=comet_logger)

The CometLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Neptune.ai
----------------------------------------------
`Neptune.ai <https://neptune.ai/>`_ is a third-party logger.
To use Neptune.ai as your logger do the following.

.. note:: See: :ref:`neptune` docs.

.. code-block:: python

   from pytorch_lightning.loggers import NeptuneLogger

    neptune_logger = NeptuneLogger(
        project_name="USER_NAME/PROJECT_NAME",
        experiment_name="default", # Optional,
        params={"max_epochs": 10}, # Optional,
        tags=["pytorch-lightning","mlp"] # Optional,
    )
   trainer = Trainer(logger=neptune_logger)

The Neptune.ai is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)


Wandb support
----------------------------------------------
`Wandb <https://www.wandb.com/>`_ is a third-party logger.
To use Wandb as your logger do the following.

.. note:: See: :ref:`wandb` docs

.. code-block:: python

   from pytorch_lightning.loggers import WandbLogger

   wandb_logger = WandbLogger()
   trainer = Trainer(logger=wandb_logger)

The Wandb logger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Save a snapshot of all hyperparameters
----------------------------------------------
When training a model, it's useful to know what hyperparams went into that model.
When Lightning creates a checkpoint, it stores a key "hparams" with the hyperparams.

.. code-block:: python

   lightning_checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
   hyperparams = lightning_checkpoint['hparams']

Some loggers also allow logging the hyperparams used in the experiment. For instance,
when using the TestTubeLogger or the TensorBoardLogger, all hyperparams will show
in the `hparams tab <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>`_.


Snapshot code for a training run
----------------------------------------------
Loggers  also allow you to snapshot a copy of the code used in this experiment.
For example, TestTubeLogger does this with a flag:

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

   logger = TestTubeLogger(create_git_tag=True)

Write logs file to csv every k batches
----------------------------------------------
Writing to a logger  can be expensive. In Lightning you can set the interval at which you
want to log using this trainer flag.

.. note:: See: :ref:`trainer`

.. code-block:: python

   k = 100
   Trainer(log_save_interval=k)

Hooks
=======
This is the order in which lightning calls the hooks. You can override each for custom behavior.

Training set-up
--------------------
- init_ddp_connection
- init_optimizers
- configure_apex
- configure_ddp
- get_train_dataloader
- get_test_dataloaders
- get_val_dataloaders
- summarize
- restore_weights

Training loop
--------------------

- on_epoch_start
- on_batch_start
- tbptt_split_batch
- training_step
- training_end (optional)
- backward
- on_after_backward
- optimizer.step()
- on_batch_end
- on_epoch_end

Validation loop
--------------------

- model.zero_grad()
- model.eval()
- torch.set_grad_enabled(False)
- validation_step
- validation_end
- model.train()
- torch.set_grad_enabled(True)
- on_post_performance_check

Test loop
------------

- model.zero_grad()
- model.eval()
- torch.set_grad_enabled(False)
- test_step
- test_end
- model.train()
- torch.set_grad_enabled(True)
- on_post_performance_check

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

