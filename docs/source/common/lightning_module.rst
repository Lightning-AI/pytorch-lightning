.. role:: hidden
    :class: hidden-section

.. _lightning_module:

LightningModule
===============
A :class:`~LightningModule` organizes your PyTorch code into 5 sections

- Computations (init).
- Train loop (training_step)
- Validation loop (validation_step)
- Test loop (test_step)
- Optimizers (configure_optimizers)

|

.. raw:: html

    <video width="100%" max-width="400px" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_mod_vid.m4v"></video>

|

Notice a few things.

1.  It's the SAME code.
2.  The PyTorch code IS NOT abstracted - just organized.
3.  All the other code that's not in the :class:`~LightningModule`
    has been automated for you by the trainer.

|

    .. code-block:: python

        net = Net()
        trainer = Trainer()
        trainer.fit(net)

4.  There are no .cuda() or .to() calls... Lightning does these for you.

|

    .. code-block:: python

        # don't do in lightning
        x = torch.Tensor(2, 3)
        x = x.cuda()
        x = x.to(device)

        # do this instead
        x = x  # leave it alone!

        # or to init a new tensor
        new_x = torch.Tensor(2, 3)
        new_x = new_x.type_as(x)

5.  Lightning by default handles the distributed sampler for you.

|

    .. code-block:: python

        # Don't do in Lightning...
        data = MNIST(...)
        sampler = DistributedSampler(data)
        DataLoader(data, sampler=sampler)

        # do this instead
        data = MNIST(...)
        DataLoader(data)

6.  A :class:`~LightningModule` is a :class:`torch.nn.Module` but with added functionality. Use it as such!

|

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
    ...         self.l1 = nn.Linear(28 * 28, 10)
    ...
    ...     def forward(self, x):
    ...         return torch.relu(self.l1(x.view(x.size(0), -1)))
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         loss = F.cross_entropy(y_hat, y)
    ...         return loss
    ...
    ...     def configure_optimizers(self):
    ...         return torch.optim.Adam(self.parameters(), lr=0.02)

Which you can train by doing:

.. code-block:: python

    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))
    trainer = pl.Trainer()
    model = LitModel()

    trainer.fit(model, train_loader)

The LightningModule has many convenience methods, but the core ones you need to know about are:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Name
     - Description
   * - init
     - Define computations here
   * - forward
     - Use for inference only (separate from training_step)
   * - training_step
     - the full training loop
   * - validation_step
     - the full validation loop
   * - test_step
     - the full test loop
   * - configure_optimizers
     - define optimizers and LR schedulers

----------

Training
--------

Training loop
^^^^^^^^^^^^^
To add a training loop use the `training_step` method

.. code-block:: python

    class LitClassifier(pl.LightningModule):

         def __init__(self, model):
             super().__init__()
             self.model = model

         def training_step(self, batch, batch_idx):
             x, y = batch
             y_hat = self.model(x)
             loss = F.cross_entropy(y_hat, y)
             return loss

Under the hood, Lightning does the following (pseudocode):

.. code-block:: python

    # put model in train mode
    model.train()
    torch.set_grad_enabled(True)

    losses = []
    for batch in train_dataloader:
        # forward
        loss = training_step(batch)
        losses.append(loss.detach())

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()


Training epoch-level metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to calculate epoch-level metrics and log them, use the `.log` method

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)

         # logs metrics for each training_step,
         # and the average across the epoch, to the progress bar and logger
         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
         return loss

The `.log` object automatically reduces the requested metrics across the full epoch.
Here's the pseudocode of what it does under the hood:

.. code-block:: python

    outs = []
    for batch in train_dataloader:
        # forward
        out = training_step(val_batch)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

    epoch_metric = torch.mean(torch.stack([x['train_loss'] for x in outs]))

Train epoch-level operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need to do something with all the outputs of each `training_step`, override `training_epoch_end` yourself.

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         preds = ...
         return {'loss': loss, 'other_stuff': preds}

     def training_epoch_end(self, training_step_outputs):
        for pred in training_step_outputs:
            # do something

The matching pseudocode is:

.. code-block:: python

    outs = []
    for batch in train_dataloader:
        # forward
        out = training_step(val_batch)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

    training_epoch_end(outs)

Training with DataParallel
~~~~~~~~~~~~~~~~~~~~~~~~~~
When training using a `accelerator` that splits data from each batch across GPUs, sometimes you might
need to aggregate them on the master GPU for processing (dp, or ddp2).

In this case, implement the `training_step_end` method

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred = ...
         return {'loss': loss, 'pred': pred}

     def training_step_end(self, batch_parts):
         gpu_0_prediction = batch_parts[0]['pred']
         gpu_1_prediction = batch_parts[1]['pred']

         # do something with both outputs
         return (batch_parts[0]['loss'] + batch_parts[1]['loss']) / 2

     def training_epoch_end(self, training_step_outputs):
        for out in training_step_outputs:
            # do something with preds

The full pseudocode that lighting does under the hood is:

.. code-block:: python

    outs = []
    for train_batch in train_dataloader:
        batches = split_batch(train_batch)
        dp_outs = []
        for sub_batch in batches:
            # 1
            dp_out = training_step(sub_batch)
            dp_outs.append(dp_out)

        # 2
        out = training_step_end(dp_outs)
        outs.append(out)

    # do something with the outputs for all batches
    # 3
    training_epoch_end(outs)

------------------

Validation loop
^^^^^^^^^^^^^^^
To add a validation loop, override the `validation_step` method of the :class:`~LightningModule`:

.. code-block:: python

    class LitModel(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('val_loss', loss)

Under the hood, Lightning does the following:

.. code-block:: python

    # ...
    for batch in train_dataloader:
        loss = model.training_step()
        loss.backward()
        # ...

        if validate_at_some_point:
            # disable grads + batchnorm + dropout
            torch.set_grad_enabled(False)
            model.eval()

            # ----------------- VAL LOOP ---------------
            for val_batch in model.val_dataloader:
                val_out = model.validation_step(val_batch)
            # ----------------- VAL LOOP ---------------

            # enable grads + batchnorm + dropout
            torch.set_grad_enabled(True)
            model.train()

Validation epoch-level metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need to do something with all the outputs of each `validation_step`, override `validation_epoch_end`.

.. code-block:: python

     def validation_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred =  ...
         return pred

     def validation_epoch_end(self, validation_step_outputs):
        for pred in validation_step_outputs:
            # do something with a pred

Validating with DataParallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When training using a `accelerator` that splits data from each batch across GPUs, sometimes you might
need to aggregate them on the master GPU for processing (dp, or ddp2).

In this case, implement the `validation_step_end` method

.. code-block:: python

     def validation_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred = ...
         return {'loss': loss, 'pred': pred}

     def validation_step_end(self, batch_parts):
         gpu_0_prediction = batch_parts.pred[0]['pred']
         gpu_1_prediction = batch_parts.pred[1]['pred']

         # do something with both outputs
         return (batch_parts[0]['loss'] + batch_parts[1]['loss']) / 2

     def validation_epoch_end(self, validation_step_outputs):
        for out in validation_step_outputs:
            # do something with preds

The full pseudocode that lighting does under the hood is:

.. code-block:: python

    outs = []
    for batch in dataloader:
        batches = split_batch(batch)
        dp_outs = []
        for sub_batch in batches:
            # 1
            dp_out = validation_step(sub_batch)
            dp_outs.append(dp_out)

        # 2
        out = validation_step_end(dp_outs)
        outs.append(out)

    # do something with the outputs for all batches
    # 3
    validation_epoch_end(outs)

----------------

Test loop
^^^^^^^^^
The process for adding a test loop is the same as the process for adding a validation loop. Please refer to
the section above for details.

The only difference is that the test loop is only called when `.test()` is used:

.. code-block:: python

    model = Model()
    trainer = Trainer()
    trainer.fit()

    # automatically loads the best weights for you
    trainer.test(model)

There are two ways to call `test()`:

.. code-block:: python

    # call after training
    trainer = Trainer()
    trainer.fit(model)

    # automatically auto-loads the best weights
    trainer.test(test_dataloaders=test_dataloader)

    # or call with pretrained model
    model = MyLightningModule.load_from_checkpoint(PATH)
    trainer = Trainer()
    trainer.test(model, test_dataloaders=test_dataloader)

----------

Inference
---------
For research, LightningModules are best structured as systems.

.. code-block:: python

    import pytorch_lightning as pl
    import torch
    from torch import nn

    class Autoencoder(pl.LightningModule):

         def __init__(self, latent_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 28 * 28))

         def training_step(self, batch, batch_idx):
            x, _ = batch

            # encode
            x = x.view(x.size(0), -1)
            z = self.encoder(x)

            # decode
            recons = self.decoder(z)

            # reconstruction
            reconstruction_loss = nn.functional.mse_loss(recons, x)
            return reconstruction_loss

         def validation_step(self, batch, batch_idx):
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            recons = self.decoder(z)
            reconstruction_loss = nn.functional.mse_loss(recons, x)
            self.log('val_reconstruction', reconstruction_loss)

         def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0002)

Which can be trained like this:

.. code-block:: python

    autoencoder = Autoencoder()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(autoencoder, train_dataloader, val_dataloader)

This simple model generates examples that look like this (the encoders and decoders are too weak)

.. figure:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/ae_docs.png
    :width: 300

The methods above are part of the lightning interface:

- training_step
- validation_step
- test_step
- configure_optimizers

Note that in this case, the train loop and val loop are exactly the same. We can of course reuse this code.

.. code-block:: python

    class Autoencoder(pl.LightningModule):

         def __init__(self, latent_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 28 * 28))

         def training_step(self, batch, batch_idx):
            loss = self.shared_step(batch)

            return loss

         def validation_step(self, batch, batch_idx):
            loss = self.shared_step(batch)
            self.log('val_loss', loss)

         def shared_step(self, batch):
            x, _ = batch

            # encode
            x = x.view(x.size(0), -1)
            z = self.encoder(x)

            # decode
            recons = self.decoder(z)

            # loss
            return nn.functional.mse_loss(recons, x)

         def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0002)

We create a new method called `shared_step` that all loops can use. This method name is arbitrary and NOT reserved.

Inference in research
^^^^^^^^^^^^^^^^^^^^^
In the case where we want to perform inference with the system we can add a `forward` method to the LightningModule.

.. code-block:: python

    class Autoencoder(pl.LightningModule):
        def forward(self, x):
            return self.decoder(x)

The advantage of adding a forward is that in complex systems, you can do a much more involved inference procedure,
such as text generation:

.. code-block:: python

    class Seq2Seq(pl.LightningModule):

        def forward(self, x):
            embeddings = self(x)
            hidden_states = self.encoder(embeddings)
            for h in hidden_states:
                # decode
                ...
            return decoded

Inference in production
^^^^^^^^^^^^^^^^^^^^^^^
For cases like production, you might want to iterate different models inside a LightningModule.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.metrics import functional as FM

    class ClassificationTask(pl.LightningModule):

         def __init__(self, model):
             super().__init__()
             self.model = model

         def training_step(self, batch, batch_idx):
             x, y = batch
             y_hat = self.model(x)
             loss = F.cross_entropy(y_hat, y)
             return loss

         def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            acc = FM.accuracy(y_hat, y)

            metrics = {'val_acc': acc, 'val_loss': loss}
            self.log_dict(metrics)
            return metrics

         def test_step(self, batch, batch_idx):
            metrics = self.validation_step(batch, batch_idx)
            metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
            self.log_dict(metrics)

         def configure_optimizers(self):
             return torch.optim.Adam(self.model.parameters(), lr=0.02)

Then pass in any arbitrary model to be fit with this task

.. code-block:: python

    for model in [resnet50(), vgg16(), BidirectionalRNN()]:
        task = ClassificationTask(model)

        trainer = Trainer(gpus=2)
        trainer.fit(task, train_dataloader, val_dataloader)

Tasks can be arbitrarily complex such as implementing GAN training, self-supervised or even RL.

.. code-block:: python

    class GANTask(pl.LightningModule):

         def __init__(self, generator, discriminator):
             super().__init__()
             self.generator = generator
             self.discriminator = discriminator
         ...

When used like this, the model can be separated from the Task and thus used in production without needing to keep it in
a `LightningModule`.

- You can export to onnx.
- Or trace using Jit.
- or run in the python runtime.

.. code-block:: python

        task = ClassificationTask(model)

        trainer = Trainer(gpus=2)
        trainer.fit(task, train_dataloader, val_dataloader)

        # use model after training or load weights and drop into the production system
        model.eval()
        y_hat = model(x)

-----------

LightningModule API
-------------------

Methods
^^^^^^^

configure_callbacks
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.configure_callbacks
    :noindex:

configure_optimizers
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    :noindex:

forward
~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.forward
    :noindex:

freeze
~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.freeze
    :noindex:

log
~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.log
    :noindex:

log_dict
~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.log_dict
    :noindex:

manual_backward
~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.manual_backward
    :noindex:

print
~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.print
    :noindex:

predict_step
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.predict_step
    :noindex:

save_hyperparameters
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.save_hyperparameters
    :noindex:

test_step
~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.test_step
    :noindex:

test_step_end
~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.test_step_end
    :noindex:

test_epoch_end
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.test_epoch_end
    :noindex:

to_onnx
~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.to_onnx
    :noindex:

to_torchscript
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.to_torchscript
    :noindex:

training_step
~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.training_step
    :noindex:

training_step_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.training_step_end
    :noindex:

training_epoch_end
~~~~~~~~~~~~~~~~~~
.. automethod:: pytorch_lightning.core.lightning.LightningModule.training_epoch_end
    :noindex:

unfreeze
~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.unfreeze
    :noindex:

validation_step
~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.validation_step
    :noindex:

validation_step_end
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.validation_step_end
    :noindex:

validation_epoch_end
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.validation_epoch_end
    :noindex:

write_prediction
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.write_prediction
    :noindex:

write_prediction_dict
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.write_prediction_dict
    :noindex:

------------

Properties
^^^^^^^^^^
These are properties available in a LightningModule.

-----------

current_epoch
~~~~~~~~~~~~~
The current epoch

.. code-block:: python

    def training_step(...):
        if self.current_epoch == 0:

-------------

device
~~~~~~
The device the module is on. Use it to keep your code device agnostic

.. code-block:: python

    def training_step(...):
        z = torch.rand(2, 3, device=self.device)

-------------

global_rank
~~~~~~~~~~~
The global_rank of this LightningModule. Lightning saves logs, weights etc only from global_rank = 0. You
normally do not need to use this property

Global rank refers to the index of that GPU across ALL GPUs. For example, if using 10 machines, each with 4 GPUs,
the 4th GPU on the 10th machine has global_rank = 39

-------------

global_step
~~~~~~~~~~~
The current step (does not reset each epoch)

.. code-block:: python

    def training_step(...):
        self.logger.experiment.log_image(..., step=self.global_step)

-------------

hparams
~~~~~~~
The arguments saved by calling ``save_hyperparameters`` passed through ``__init__()``
 could be accessed by the ``hparams`` attribute.

.. code-block:: python

    def __init__(self, learning_rate):
        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

--------------

logger
~~~~~~
The current logger being used (tensorboard or other supported logger)

.. code-block:: python

    def training_step(...):
        # the generic logger (same no matter if tensorboard or other supported logger)
        self.logger

        # the particular logger
        tensorboard_logger = self.logger.experiment

--------------

local_rank
~~~~~~~~~~~
The local_rank of this LightningModule. Lightning saves logs, weights etc only from global_rank = 0. You
normally do not need to use this property

Local rank refers to the rank on that machine. For example, if using 10 machines, the GPU at index 0 on each machine
has local_rank = 0.


-----------

precision
~~~~~~~~~
The type of precision used:

.. code-block:: python

    def training_step(...):
        if self.precision == 16:

------------

trainer
~~~~~~~
Pointer to the trainer

.. code-block:: python

    def training_step(...):
        max_steps = self.trainer.max_steps
        any_flag = self.trainer.any_flag

------------

use_amp
~~~~~~~
True if using Automatic Mixed Precision (AMP)

--------------

automatic_optimization
~~~~~~~~~~~~~~~~~~~~~~
When set to ``False``, Lightning does not automate the optimization process. This means you are responsible for handling
your optimizers. However, we do take care of precision and any accelerators used.

See :ref:`manual optimization<common/optimizers:Manual optimization>` for details.

.. code-block:: python

    def __init__(self):
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)

        loss = ...
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

This is recommended only if using 2+ optimizers AND if you know how to perform the optimization procedure properly. Note
that automatic optimization can still be used with multiple optimizers by relying on the ``optimizer_idx`` parameter.
Manual optimization is most useful for research topics like reinforcement learning, sparse coding, and GAN research.

.. code-block:: python

    def __init__(self):
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

        gen_loss = ...
        opt_a.zero_grad()
        self.manual_backward(gen_loss)
        opt_a.step()

        disc_loss = ...
        opt_b.zero_grad()
        self.manual_backward(disc_loss)
        opt_b.step()

--------------

example_input_array
~~~~~~~~~~~~~~~~~~~
Set and access example_input_array which is basically a single batch.

.. code-block:: python

    def __init__(self):
        self.example_input_array = ...
        self.generator = ...

    def on_train_epoch_end(...):
        # generate some images using the example_input_array
        gen_images = self.generator(self.example_input_array)

--------------

datamodule
~~~~~~~~~~
Set or access your datamodule.

.. code-block:: python

    def configure_optimizers(self):
        num_training_samples = len(self.trainer.datamodule.train_dataloader())
        ...

--------------

model_size
~~~~~~~~~~
Get the model file size (in megabytes) using ``self.model_size`` inside LightningModule.

--------------

truncated_bptt_steps
^^^^^^^^^^^^^^^^^^^^

Truncated back prop breaks performs backprop every k steps of
a much longer sequence.

If this is enabled, your batches will automatically get truncated
and the trainer will apply Truncated Backprop to it.

(`Williams et al. "An efficient gradient-based algorithm for on-line training of
recurrent network trajectories."
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf>`_)

`Tutorial <https://d2l.ai/chapter_recurrent-neural-networks/bptt.html>`_

.. testcode:: python

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

Lightning takes care to split your batch along the time-dimension.

.. code-block:: python

    # we use the second as the time dimension
    # (batch, time, ...)
    sub_batch = batch[0, 0:t, ...]

To modify how the batch is split,
override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`:

.. testcode:: python

    class LitMNIST(LightningModule):
        def tbptt_split_batch(self, batch, split_size):
            # do your own splitting on the batch
            return splits

--------------

Hooks
^^^^^
This is the pseudocode to describe the structure of :meth:`~pytorch_lightning.trainer.Trainer.fit`.
The inputs and outputs of each function are not represented for simplicity. Please check each function's API reference
for more information.

.. code-block:: python

    def fit(...):
        if global_rank == 0:
            # prepare data is called on GLOBAL_ZERO only
            prepare_data()

        configure_callbacks()

        with parallel(devices):
            # devices can be GPUs, TPUs, ...
            train_on_device(model)

    def train_on_device(model):
        # called PER DEVICE
        on_fit_start()
        setup('fit')
        configure_optimizers()

        on_pretrain_routine_start()
        on_pretrain_routine_end()

        # the sanity check runs here

        on_train_start()
        for epoch in epochs:
            train_loop()
        on_train_end()

        on_fit_end()
        teardown('fit')

    def train_loop():
        on_epoch_start()
        on_train_epoch_start()

        for batch in train_dataloader():
            on_train_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            training_step()

            on_before_zero_grad()
            optimizer_zero_grad()

            backward()
            on_after_backward()

            optimizer_step()

            on_train_batch_end()

            if should_check_val:
                val_loop()
        # end training epoch
        training_epoch_end()

        on_train_epoch_end()
        on_epoch_end()

    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)

        on_validation_start()
        on_epoch_start()
        on_validation_epoch_start()

        for batch in val_dataloader():
            on_validation_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            validation_step()

            on_validation_batch_end()
        validation_epoch_end()

        on_validation_epoch_end()
        on_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)

backward
~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.backward
    :noindex:

get_progress_bar_dict
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.get_progress_bar_dict
    :noindex:

on_after_backward
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_after_backward
    :noindex:

on_before_zero_grad
~~~~~~~~~~~~~~~~~~~
.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad
    :noindex:

on_fit_start
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_fit_start
    :noindex:

on_fit_end
~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_fit_end
    :noindex:


on_load_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint
    :noindex:

on_save_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint
    :noindex:

on_train_start
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_start
    :noindex:

on_train_end
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_end
    :noindex:

on_validation_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_start
    :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_end
    :noindex:

on_pretrain_routine_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_pretrain_routine_start
    :noindex:

on_pretrain_routine_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_pretrain_routine_end
    :noindex:

on_test_batch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_batch_start
    :noindex:

on_test_batch_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_batch_end
    :noindex:

on_test_epoch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_epoch_start
    :noindex:

on_test_epoch_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_epoch_end
    :noindex:

on_test_end
~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_end
    :noindex:

on_train_batch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_batch_start
    :noindex:

on_train_batch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_batch_end
    :noindex:

on_epoch_start
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_epoch_start
    :noindex:

on_epoch_end
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_epoch_end
    :noindex:

on_train_epoch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_epoch_start
    :noindex:

on_train_epoch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_train_epoch_end
    :noindex:

on_validation_batch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_batch_start
    :noindex:

on_validation_batch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_batch_end
    :noindex:

on_validation_epoch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_epoch_end
    :noindex:

on_post_move_to_device
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_post_move_to_device
    :noindex:

on_validation_model_eval
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval
    :noindex:

on_validation_model_train
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_validation_model_train
    :noindex:

on_test_model_eval
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_model_eval
    :noindex:

on_test_model_train
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.ModelHooks.on_test_model_train
    :noindex:

optimizer_step
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.optimizer_step
    :noindex:

optimizer_zero_grad
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.optimizer_zero_grad
    :noindex:

prepare_data
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.prepare_data
    :noindex:

setup
~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.setup
    :noindex:

tbptt_split_batch
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch
    :noindex:

teardown
~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.teardown
    :noindex:

train_dataloader
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.train_dataloader
    :noindex:

val_dataloader
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.val_dataloader
    :noindex:

test_dataloader
~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.test_dataloader
    :noindex:

transfer_batch_to_device
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.transfer_batch_to_device
    :noindex:

on_before_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.on_before_batch_transfer
    :noindex:

on_after_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.hooks.DataHooks.on_after_batch_transfer
    :noindex:
