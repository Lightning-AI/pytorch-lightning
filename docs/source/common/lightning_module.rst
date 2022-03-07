.. role:: hidden
    :class: hidden-section

.. _lightning_module:

###############
LightningModule
###############

A :class:`~LightningModule` organizes your PyTorch code into 6 sections:

- Computations (init).
- Train Loop (training_step)
- Validation Loop (validation_step)
- Test Loop (test_step)
- Prediction Loop (predict_step)
- Optimizers and LR Schedulers (configure_optimizers)

|

.. raw:: html

    <video width="100%" max-width="400px" controls autoplay muted playsinline src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_mod_vid.m4v"></video>

|

Notice a few things.

1.  It is the SAME code.
2.  The PyTorch code IS NOT abstracted - just organized.
3.  All the other code that's not in the :class:`~LightningModule`
    has been automated for you by the Trainer.

|

    .. code-block:: python

        net = Net()
        trainer = Trainer()
        trainer.fit(net)

4.  There are no ``.cuda()`` or ``.to(device)`` calls required. Lightning does these for you.

|

    .. code-block:: python

        # don't do in Lightning
        x = torch.Tensor(2, 3)
        x = x.cuda()
        x = x.to(device)

        # do this instead
        x = x  # leave it alone!

        # or to init a new tensor
        new_x = torch.Tensor(2, 3)
        new_x = new_x.type_as(x)

5. When running under a distributed strategy, Lightning handles the distributed sampler for you by default.

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
(and let's be real, you probably should do anyway).

------------

***************
Starter Example
***************

Here are the only required methods.

.. code-block:: python

    import pytorch_lightning as pl
    import torch.nn as nn
    import torch.nn.functional as F


    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

Which you can train by doing:

.. code-block:: python

    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))
    trainer = pl.Trainer(max_epochs=1)
    model = LitModel()

    trainer.fit(model, train_dataloaders=train_loader)

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
     - the complete training loop
   * - validation_step
     - the complete validation loop
   * - test_step
     - the complete test loop
   * - predict_step
     - the complete prediction loop
   * - configure_optimizers
     - define optimizers and LR schedulers

----------

********
Training
********

Training Loop
=============

To activate the training loop, override the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` method.

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

    # put model in train mode and enable gradient calculation
    model.train()
    torch.set_grad_enabled(True)

    outs = []
    for batch_idx, batch in enumerate(train_dataloader):
        loss = training_step(batch, batch_idx)
        outs.append(loss.detach())

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()


Train Epoch-level Metrics
=========================

If you want to calculate epoch-level metrics and log them, use :meth:`~pytorch_lightning.core.lightning.LightningModule.log`.

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)

         # logs metrics for each training_step,
         # and the average across the epoch, to the progress bar and logger
         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
         return loss

The :meth:`~pytorch_lightning.core.lightning.LightningModule.log` object automatically reduces the
requested metrics across a complete epoch and devices. Here's the pseudocode of what it does under the hood:

.. code-block:: python

    outs = []
    for batch_idx, batch in enumerate(train_dataloader):
        # forward
        loss = training_step(batch, batch_idx)
        outs.append(loss)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

    epoch_metric = torch.mean(torch.stack([x for x in outs]))

Train Epoch-level Operations
============================

If you need to do something with all the outputs of each :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`,
override the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_epoch_end` method.

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         preds = ...
         return {"loss": loss, "other_stuff": preds}


     def training_epoch_end(self, training_step_outputs):
         all_preds = torch.stack(training_step_outputs)
         ...

The matching pseudocode is:

.. code-block:: python

    outs = []
    for batch_idx, batch in enumerate(train_dataloader):
        # forward
        loss = training_step(batch, batch_idx)
        outs.append(loss)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

    training_epoch_end(outs)

Training with DataParallel
==========================

When training using a ``strategy`` that splits data from each batch across GPUs, sometimes you might
need to aggregate them on the main GPU for processing (DP, or DDP2).

In this case, implement the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step_end`
method which will have outputs from all the devices and you can accumulate to get the effective results.

.. code-block:: python

     def training_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred = ...
         return {"loss": loss, "pred": pred}


     def training_step_end(self, batch_parts):
         # predictions from each GPU
         predictions = batch_parts["pred"]
         # losses from each GPU
         losses = batch_parts["loss"]

         gpu_0_prediction = predictions[0]
         gpu_1_prediction = predictions[1]

         # do something with both outputs
         return (losses[0] + losses[1]) / 2


     def training_epoch_end(self, training_step_outputs):
         for out in training_step_outputs:
             ...

Here is the Lightning training pseudo-code for DP:

.. code-block:: python

    outs = []
    for batch_idx, train_batch in enumerate(train_dataloader):
        batches = split_batch(train_batch)
        dp_outs = []
        for sub_batch in batches:
            # 1
            dp_out = training_step(sub_batch, batch_idx)
            dp_outs.append(dp_out)

        # 2
        out = training_step_end(dp_outs)
        outs.append(out)

    # do something with the outputs for all batches
    # 3
    training_epoch_end(outs)

------------------

**********
Validation
**********

Validation Loop
===============

To activate the validation loop while training, override the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` method.

.. code-block:: python

    class LitModel(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", loss)

Under the hood, Lightning does the following (pseudocode):

.. code-block:: python

    # ...
    for batch_idx, batch in enumerate(train_dataloader):
        loss = model.training_step(batch, batch_idx)
        loss.backward()
        # ...

        if validate_at_some_point:
            # disable grads + batchnorm + dropout
            torch.set_grad_enabled(False)
            model.eval()

            # ----------------- VAL LOOP ---------------
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_out = model.validation_step(val_batch, val_batch_idx)
            # ----------------- VAL LOOP ---------------

            # enable grads + batchnorm + dropout
            torch.set_grad_enabled(True)
            model.train()

You can also run just the validation loop on your validation dataloaders by overriding :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`
and calling :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`.

.. code-block:: python

    model = Model()
    trainer = Trainer()
    trainer.validate(model)

.. note::

    It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once.
    This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a
    multi-device setting, samples could occur duplicated when :class:`~torch.utils.data.distributed.DistributedSampler`
    is used, for eg. with ``strategy="ddp"``. It replicates some samples on some devices to make sure all devices have
    same batch size in case of uneven inputs.


Validation Epoch-level Metrics
==============================

If you need to do something with all the outputs of each :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`,
override the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end` method. Note that this method is called before :meth:`~pytorch_lightning.core.lightning.LightningModule.training_epoch_end`.

.. code-block:: python

     def validation_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred = ...
         return pred


     def validation_epoch_end(self, validation_step_outputs):
         all_preds = torch.stack(validation_step_outputs)
         ...

Validating with DataParallel
============================

When training using a ``strategy`` that splits data from each batch across GPUs, sometimes you might
need to aggregate them on the main GPU for processing (DP, or DDP2).

In this case, implement the :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step_end`
method which will have outputs from all the devices and you can accumulate to get the effective results.

.. code-block:: python

     def validation_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self.model(x)
         loss = F.cross_entropy(y_hat, y)
         pred = ...
         return {"loss": loss, "pred": pred}


     def validation_step_end(self, batch_parts):
         # predictions from each GPU
         predictions = batch_parts["pred"]
         # losses from each GPU
         losses = batch_parts["loss"]

         gpu_0_prediction = predictions[0]
         gpu_1_prediction = predictions[1]

         # do something with both outputs
         return (losses[0] + losses[1]) / 2


     def validation_epoch_end(self, validation_step_outputs):
         for out in validation_step_outputs:
             ...

Here is the Lightning validation pseudo-code for DP:

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

*******
Testing
*******

Test Loop
=========

The process for enabling a test loop is the same as the process for enabling a validation loop. Please refer to
the section above for details. For this you need to override the :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step` method.

The only difference is that the test loop is only called when :meth:`~pytorch_lightning.trainer.trainer.Trainer.test` is used.

.. code-block:: python

    model = Model()
    trainer = Trainer()
    trainer.fit(model)

    # automatically loads the best weights for you
    trainer.test(model)

There are two ways to call ``test()``:

.. code-block:: python

    # call after training
    trainer = Trainer()
    trainer.fit(model)

    # automatically auto-loads the best weights from the previous run
    trainer.test(dataloaders=test_dataloader)

    # or call with pretrained model
    model = MyLightningModule.load_from_checkpoint(PATH)
    trainer = Trainer()
    trainer.test(model, dataloaders=test_dataloader)

.. note::

    It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once.
    This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a
    multi-device setting, samples could occur duplicated when :class:`~torch.utils.data.distributed.DistributedSampler`
    is used, for eg. with ``strategy="ddp"``. It replicates some samples on some devices to make sure all devices have
    same batch size in case of uneven inputs.


----------

*********
Inference
*********

Prediction Loop
===============

By default, the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` method runs the
:meth:`~pytorch_lightning.core.lightning.LightningModule.forward` method. In order to customize this behaviour,
simply override the :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` method.

For the example let's override ``predict_step`` and try out `Monte Carlo Dropout <https://arxiv.org/pdf/1506.02142.pdf>`_:

.. code-block:: python

    class LitMCdropoutModel(pl.LightningModule):
        def __init__(self, model, mc_iteration):
            super().__init__()
            self.model = model
            self.dropout = nn.Dropout()
            self.mc_iteration = mc_iteration

        def predict_step(self, batch, batch_idx):
            # enable Monte Carlo Dropout
            self.dropout.train()

            # take average of `self.mc_iteration` iterations
            pred = torch.vstack([self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]).mean(dim=0)
            return pred

Under the hood, Lightning does the following (pseudocode):

.. code-block:: python

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    model.eval()
    all_preds = []

    for batch_idx, batch in enumerate(predict_dataloader):
        pred = model.predict_step(batch, batch_idx)
        all_preds.append(pred)

There are two ways to call ``predict()``:

.. code-block:: python

    # call after training
    trainer = Trainer()
    trainer.fit(model)

    # automatically auto-loads the best weights from the previous run
    predictions = trainer.predict(dataloaders=predict_dataloader)

    # or call with pretrained model
    model = MyLightningModule.load_from_checkpoint(PATH)
    trainer = Trainer()
    predictions = trainer.predict(model, dataloaders=test_dataloader)

Inference in Research
=====================

If you want to perform inference with the system, you can add a ``forward`` method to the LightningModule.

.. note:: When using forward, you are responsible to call :func:`~torch.nn.Module.eval` and use the :func:`~torch.no_grad` context manager.

.. code-block:: python

    class Autoencoder(pl.LightningModule):
        def forward(self, x):
            return self.decoder(x)


    model = Autoencoder()
    model.eval()
    with torch.no_grad():
        reconstruction = model(embedding)

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

In the case where you want to scale your inference, you should be using
:meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step`.

.. code-block:: python

    class Autoencoder(pl.LightningModule):
        def forward(self, x):
            return self.decoder(x)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            # this calls forward
            return self(batch)


    data_module = ...
    model = Autoencoder()
    trainer = Trainer(gpus=2)
    trainer.predict(model, data_module)

Inference in Production
=======================

For cases like production, you might want to iterate different models inside a LightningModule.

.. code-block:: python

    from torchmetrics.functional import accuracy


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
            loss, acc = self._shared_eval_step(batch, batch_idx)
            metrics = {"val_acc": acc, "val_loss": loss}
            self.log_dict(metrics)
            return metrics

        def test_step(self, batch, batch_idx):
            loss, acc = self._shared_eval_step(batch, batch_idx)
            metrics = {"test_acc": acc, "test_loss": loss}
            self.log_dict(metrics)
            return metrics

        def _shared_eval_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat, y)
            return loss, acc

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            x, y = batch
            y_hat = self.model(x)
            return y_hat

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.02)

Then pass in any arbitrary model to be fit with this task

.. code-block:: python

    for model in [resnet50(), vgg16(), BidirectionalRNN()]:
        task = ClassificationTask(model)

        trainer = Trainer(gpus=2)
        trainer.fit(task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

Tasks can be arbitrarily complex such as implementing GAN training, self-supervised or even RL.

.. code-block:: python

    class GANTask(pl.LightningModule):
        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator

        ...

When used like this, the model can be separated from the Task and thus used in production without needing to keep it in
a ``LightningModule``.

The following example shows how you can run inference in the Python runtime:

.. code-block:: python

    task = ClassificationTask(model)
    trainer = Trainer(gpus=2)
    trainer.fit(task, train_dataloader, val_dataloader)
    trainer.save_checkpoint("best_model.ckpt")

    # use model after training or load weights and drop into the production system
    model = ClassificationTask.load_from_checkpoint("best_model.ckpt")
    x = ...
    model.eval()
    with torch.no_grad():
        y_hat = model(x)

Check out :ref:`Inference in Production <production_inference>` guide to learn about the possible ways to perform inference in production.


-----------


*************
Child Modules
*************

.. include:: ../common/child_modules.rst

-----------

*******************
LightningModule API
*******************


Methods
=======

all_gather
~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.all_gather
    :noindex:

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

lr_schedulers
~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.lr_schedulers
    :noindex:

manual_backward
~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.manual_backward
    :noindex:

optimizers
~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.optimizers
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

toggle_optimizer
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.toggle_optimizer
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

untoggle_optimizer
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.untoggle_optimizer
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

-----------

Properties
==========

These are properties available in a LightningModule.

current_epoch
~~~~~~~~~~~~~

The current epoch

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            ...

device
~~~~~~

The device the module is on. Use it to keep your code device agnostic.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        z = torch.rand(2, 3, device=self.device)

global_rank
~~~~~~~~~~~

The ``global_rank`` is the index of the current process across all nodes and devices.
Lightning will perform some operations such as logging, weight checkpointing only when ``global_rank=0``. You
usually do not need to use this property, but it is useful to know how to access it if needed.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.global_rank == 0:
            # do something only once across all the nodes
            self.log("global_step", self.trainer.global_step)

global_step
~~~~~~~~~~~

The current step (does not reset each epoch)

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.logger.experiment.log_image(..., step=self.global_step)

hparams
~~~~~~~

The arguments passed through ``LightningModule.__init__()`` and saved by calling
:meth:`~pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin.save_hyperparameters` could be accessed by the ``hparams`` attribute.

.. code-block:: python

    def __init__(self, learning_rate):
        self.save_hyperparameters()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

logger
~~~~~~

The current logger being used (tensorboard or other supported logger)

.. code-block:: python

    def training_step(self, batch, batch_idx):
        # the generic logger (same no matter if tensorboard or other supported logger)
        self.logger

        # the particular logger
        tensorboard_logger = self.logger.experiment

loggers
~~~~~~~

The list of loggers currently being used by the Trainer.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        # List of LightningLoggerBase objects
        loggers = self.loggers
        for logger in loggers:
            logger.log_metrics({"foo": 1.0})

local_rank
~~~~~~~~~~~

The ``global_rank`` is the index of the current process across all the devices for the current node.
You usually do not need to use this property, but it is useful to know how to access it if needed.
For example, if using 10 machines (or nodes), the GPU at index 0 on each machine has local_rank = 0.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.global_rank == 0:
            # do something only once across each node
            self.log("global_step", self.trainer.global_step)

precision
~~~~~~~~~

The type of precision used:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.precision == 16:
            ...

trainer
~~~~~~~

Pointer to the trainer

.. code-block:: python

    def training_step(self, batch, batch_idx):
        max_steps = self.trainer.max_steps
        any_flag = self.trainer.any_flag

use_amp
~~~~~~~

``True`` if using Automatic Mixed Precision (AMP)

prepare_data_per_node
~~~~~~~~~~~~~~~~~~~~~

If set to ``True`` will call ``prepare_data()`` on LOCAL_RANK=0 for every node.
If set to ``False`` will only call from NODE_RANK=0, LOCAL_RANK=0.

.. testcode::

    class LitModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.prepare_data_per_node = True

automatic_optimization
~~~~~~~~~~~~~~~~~~~~~~

When set to ``False``, Lightning does not automate the optimization process. This means you are responsible for handling
your optimizers. However, we do take care of precision and any accelerators used.

See :ref:`manual optimization<common/optimization:Manual optimization>` for details.

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

example_input_array
~~~~~~~~~~~~~~~~~~~

Set and access example_input_array, which basically represents a single batch.

.. code-block:: python

    def __init__(self):
        self.example_input_array = ...
        self.generator = ...


    def on_train_epoch_end(self):
        # generate some images using the example_input_array
        gen_images = self.generator(self.example_input_array)

model_size
~~~~~~~~~~

Get the model file size (in megabytes) using ``self.model_size`` inside LightningModule.

truncated_bptt_steps
~~~~~~~~~~~~~~~~~~~~

Truncated Backpropagation Through Time (TBPTT) performs perform backpropogation every k steps of
a much longer sequence. This is made possible by passing training batches
split along the time-dimensions into splits of size k to the
``training_step``. In order to keep the same forward propagation behavior, all
hidden states should be kept in-between each time-dimension split.


If this is enabled, your batches will automatically get truncated
and the Trainer will apply Truncated Backprop to it.

(`Williams et al. "An efficient gradient-based algorithm for on-line training of
recurrent network trajectories."
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf>`_)

`Tutorial <https://d2l.ai/chapter_recurrent-neural-networks/bptt.html>`_

.. testcode:: python

    from pytorch_lightning import LightningModule


    class MyModel(LightningModule):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            # batch_first has to be set to True
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

            ...

            # Important: This property activates truncated backpropagation through time
            # Setting this value to 2 splits the batch into sequences of size 2
            self.truncated_bptt_steps = 2

        # Truncated back-propagation through time
        def training_step(self, batch, batch_idx, hiddens):
            x, y = batch

            # the training step must be updated to accept a ``hiddens`` argument
            # hiddens are the hiddens from the previous truncated backprop step
            out, hiddens = self.lstm(x, hiddens)

            ...

            return {"loss": ..., "hiddens": hiddens}

Lightning takes care of splitting your batch along the time-dimension. It is
assumed to be the second dimension of your batches. Therefore, in the
example above, we have set ``batch_first=True``.

.. code-block:: python

    # we use the second as the time dimension
    # (batch, time, ...)
    sub_batch = batch[0, 0:t, ...]

To modify how the batch is split,
override the :meth:`pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch` method:

.. testcode:: python

    class LitMNIST(LightningModule):
        def tbptt_split_batch(self, batch, split_size):
            # do your own splitting on the batch
            return splits

--------------

.. _lightning_hooks:

Hooks
=====

This is the pseudocode to describe the structure of :meth:`~pytorch_lightning.trainer.Trainer.fit`.
The inputs and outputs of each function are not represented for simplicity. Please check each function's API reference
for more information.

.. code-block:: python

    def fit(self):
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
        setup("fit")
        configure_optimizers()

        # the sanity check runs here

        on_train_start()
        for epoch in epochs:
            fit_loop()
        on_train_end()

        on_fit_end()
        teardown("fit")


    def fit_loop():
        on_train_epoch_start()

        for batch in train_dataloader():
            on_train_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            training_step()

            on_before_zero_grad()
            optimizer_zero_grad()

            on_before_backward()
            backward()
            on_after_backward()

            on_before_optimizer_step()
            configure_gradient_clipping()
            optimizer_step()

            on_train_batch_end()

            if should_check_val:
                val_loop()
        # end training epoch
        training_epoch_end()

        on_train_epoch_end()


    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)

        on_validation_start()
        on_validation_epoch_start()

        val_outs = []
        for batch_idx, batch in enumerate(val_dataloader()):
            on_validation_batch_start(batch, batch_idx)

            batch = on_before_batch_transfer(batch)
            batch = transfer_batch_to_device(batch)
            batch = on_after_batch_transfer(batch)

            out = validation_step(batch, batch_idx)

            on_validation_batch_end(batch, batch_idx)
            val_outs.append(out)

        validation_epoch_end(val_outs)

        on_validation_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)

backward
~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.backward
    :noindex:

on_before_backward
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_before_backward
    :noindex:

on_after_backward
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_after_backward
    :noindex:

on_before_zero_grad
~~~~~~~~~~~~~~~~~~~
.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_before_zero_grad
    :noindex:

on_fit_start
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_fit_start
    :noindex:

on_fit_end
~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_fit_end
    :noindex:


on_load_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_load_checkpoint
    :noindex:

on_save_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_save_checkpoint
    :noindex:

load_from_checkpoint
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.load_from_checkpoint
    :noindex:

on_hpc_save
~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_hpc_save
    :noindex:

on_hpc_load
~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_hpc_load
    :noindex:

on_train_start
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_start
    :noindex:

on_train_end
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_end
    :noindex:

on_validation_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_start
    :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_end
    :noindex:

on_test_batch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_batch_start
    :noindex:

on_test_batch_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_batch_end
    :noindex:

on_test_epoch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_epoch_start
    :noindex:

on_test_epoch_end
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_epoch_end
    :noindex:

on_test_start
~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_start
    :noindex:

on_test_end
~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_end
    :noindex:

on_predict_batch_start
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_batch_start
    :noindex:

on_predict_batch_end
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_batch_end
    :noindex:

on_predict_epoch_start
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_epoch_start
    :noindex:

on_predict_epoch_end
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_epoch_end
    :noindex:

on_predict_start
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_start
    :noindex:

on_predict_end
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_end
    :noindex:

on_train_batch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_batch_start
    :noindex:

on_train_batch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_batch_end
    :noindex:

on_train_epoch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_epoch_start
    :noindex:

on_train_epoch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_epoch_end
    :noindex:

on_validation_batch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_batch_start
    :noindex:

on_validation_batch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_batch_end
    :noindex:

on_validation_epoch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_epoch_end
    :noindex:

on_post_move_to_device
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_post_move_to_device
    :noindex:

configure_sharded_model
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.configure_sharded_model
    :noindex:

on_validation_model_eval
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_model_eval
    :noindex:

on_validation_model_train
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_validation_model_train
    :noindex:

on_test_model_eval
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_model_eval
    :noindex:

on_test_model_train
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_model_train
    :noindex:

on_before_optimizer_step
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_before_optimizer_step
    :noindex:

configure_gradient_clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.configure_gradient_clipping
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

.. automethod:: pytorch_lightning.core.lightning.LightningModule.setup
    :noindex:

tbptt_split_batch
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.tbptt_split_batch
    :noindex:

teardown
~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.teardown
    :noindex:

train_dataloader
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.train_dataloader
    :noindex:

val_dataloader
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.val_dataloader
    :noindex:

test_dataloader
~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.test_dataloader
    :noindex:

predict_dataloader
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.predict_dataloader
    :noindex:

on_train_dataloader
~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_train_dataloader
    :noindex:

on_val_dataloader
~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_val_dataloader
    :noindex:

on_test_dataloader
~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_test_dataloader
    :noindex:

on_predict_dataloader
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_predict_dataloader
    :noindex:

transfer_batch_to_device
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.transfer_batch_to_device
    :noindex:

on_before_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_before_batch_transfer
    :noindex:

on_after_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.on_after_batch_transfer
    :noindex:

add_to_queue
~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.add_to_queue
    :noindex:

get_from_queue
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.core.lightning.LightningModule.get_from_queue
    :noindex:
