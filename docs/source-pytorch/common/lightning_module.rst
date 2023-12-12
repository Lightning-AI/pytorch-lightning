.. role:: hidden
    :class: hidden-section

.. _lightning_module:

###############
LightningModule
###############

A :class:`~lightning.pytorch.core.LightningModule` organizes your PyTorch code into 6 sections:

- Initialization (``__init__`` and :meth:`~lightning.pytorch.core.hooks.ModelHooks.setup`).
- Train Loop (:meth:`~lightning.pytorch.core.LightningModule.training_step`)
- Validation Loop (:meth:`~lightning.pytorch.core.LightningModule.validation_step`)
- Test Loop (:meth:`~lightning.pytorch.core.LightningModule.test_step`)
- Prediction Loop (:meth:`~lightning.pytorch.core.LightningModule.predict_step`)
- Optimizers and LR Schedulers (:meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`)

When you convert to use Lightning, the code IS NOT abstracted - just organized.
All the other code that's not in the :class:`~lightning.pytorch.core.LightningModule`
has been automated for you by the :class:`~lightning.pytorch.trainer.trainer.Trainer`.

|

    .. code-block:: python

        net = MyLightningModuleNet()
        trainer = Trainer()
        trainer.fit(net)

There are no ``.cuda()`` or ``.to(device)`` calls required. Lightning does these for you.

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
        new_x = new_x.to(x)

When running under a distributed strategy, Lightning handles the distributed sampler for you by default.

|

    .. code-block:: python

        # Don't do in Lightning...
        data = MNIST(...)
        sampler = DistributedSampler(data)
        DataLoader(data, sampler=sampler)

        # do this instead
        data = MNIST(...)
        DataLoader(data)

A :class:`~lightning.pytorch.core.LightningModule` is a :class:`torch.nn.Module` but with added functionality. Use it as such!

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

    import lightning as L
    import torch

    from lightning.pytorch.demos import Transformer


    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)

        def forward(self, inputs, target):
            return self.model(inputs, target)

        def training_step(self, batch, batch_idx):
            inputs, target = batch
            output = self(inputs, target)
            loss = torch.nn.functional.nll_loss(output, target.view(-1))
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.1)

Which you can train by doing:

.. code-block:: python

    from lightning.pytorch.demos import WikiText2
    from torch.utils.data import DataLoader

    dataset = WikiText2()
    dataloader = DataLoader(dataset)
    model = LightningTransformer(vocab_size=dataset.vocab_size)

    trainer = L.Trainer(fast_dev_run=100)
    trainer.fit(model=model, train_dataloaders=dataloader)

The LightningModule has many convenient methods, but the core ones you need to know about are:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Name
     - Description
   * - ``__init__`` and :meth:`~lightning.pytorch.core.hooks.ModelHooks.setup`
     - Define initialization here
   * - :meth:`~lightning.pytorch.core.LightningModule.forward`
     - To run data through your model only (separate from ``training_step``)
   * - :meth:`~lightning.pytorch.core.LightningModule.training_step`
     - the complete training step
   * - :meth:`~lightning.pytorch.core.LightningModule.validation_step`
     - the complete validation step
   * - :meth:`~lightning.pytorch.core.LightningModule.test_step`
     - the complete test step
   * - :meth:`~lightning.pytorch.core.LightningModule.predict_step`
     - the complete prediction step
   * - :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
     - define optimizers and LR schedulers

----------

********
Training
********

Training Loop
=============

To activate the training loop, override the :meth:`~lightning.pytorch.core.LightningModule.training_step` method.

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)

        def training_step(self, batch, batch_idx):
            inputs, target = batch
            output = self.model(inputs, target)
            loss = torch.nn.functional.nll_loss(output, target.view(-1))
            return loss

Under the hood, Lightning does the following (pseudocode):

.. code-block:: python

    # enable gradient calculation
    torch.set_grad_enabled(True)

    for batch_idx, batch in enumerate(train_dataloader):
        loss = training_step(batch, batch_idx)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()


Train Epoch-level Metrics
=========================

If you want to calculate epoch-level metrics and log them, use :meth:`~lightning.pytorch.core.LightningModule.log`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

The :meth:`~lightning.pytorch.core.LightningModule.log` method automatically reduces the
requested metrics across a complete epoch and devices. Here's the pseudocode of what it does under the hood:

.. code-block:: python

    outs = []
    for batch_idx, batch in enumerate(train_dataloader):
        # forward
        loss = training_step(batch, batch_idx)
        outs.append(loss.detach())

        # clear gradients
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update parameters
        optimizer.step()

    # note: in reality, we do this incrementally, instead of keeping all outputs in memory
    epoch_metric = torch.mean(torch.stack(outs))

Train Epoch-level Operations
============================

In the case that you need to make use of all the outputs from each :meth:`~lightning.pytorch.LightningModule.training_step`,
override the :meth:`~lightning.pytorch.LightningModule.on_train_epoch_end` method.

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)
            self.training_step_outputs = []

        def training_step(self, batch, batch_idx):
            inputs, target = batch
            output = self.model(inputs, target)
            loss = torch.nn.functional.nll_loss(output, target.view(-1))
            preds = ...
            self.training_step_outputs.append(preds)
            return loss

        def on_train_epoch_end(self):
            all_preds = torch.stack(self.training_step_outputs)
            # do something with all preds
            ...
            self.training_step_outputs.clear()  # free memory


------------------

**********
Validation
**********

Validation Loop
===============

To activate the validation loop while training, override the :meth:`~lightning.pytorch.core.LightningModule.validation_step` method.

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def validation_step(self, batch, batch_idx):
            inputs, target = batch
            output = self.model(inputs, target)
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

You can also run just the validation loop on your validation dataloaders by overriding :meth:`~lightning.pytorch.core.LightningModule.validation_step`
and calling :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`.

.. code-block:: python

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    trainer = L.Trainer()
    trainer.validate(model)

.. note::

    It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once.
    This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a
    multi-device setting, samples could occur duplicated when :class:`~torch.utils.data.distributed.DistributedSampler`
    is used, for eg. with ``strategy="ddp"``. It replicates some samples on some devices to make sure all devices have
    same batch size in case of uneven inputs.


Validation Epoch-level Metrics
==============================

In the case that you need to make use of all the outputs from each :meth:`~lightning.pytorch.LightningModule.validation_step`,
override the :meth:`~lightning.pytorch.LightningModule.on_validation_epoch_end` method.
Note that this method is called before :meth:`~lightning.pytorch.LightningModule.on_train_epoch_end`.

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)
            self.validation_step_outputs = []

        def validation_step(self, batch, batch_idx):
            x, y = batch
            inputs, target = batch
            output = self.model(inputs, target)
            loss = torch.nn.functional.nll_loss(output, target.view(-1))
            pred = ...
            self.validation_step_outputs.append(pred)
            return pred

        def on_validation_epoch_end(self):
            all_preds = torch.stack(self.validation_step_outputs)
            # do something with all preds
            ...
            self.validation_step_outputs.clear()  # free memory

----------------

*******
Testing
*******

Test Loop
=========

The process for enabling a test loop is the same as the process for enabling a validation loop. Please refer to
the section above for details. For this you need to override the :meth:`~lightning.pytorch.core.LightningModule.test_step` method.

The only difference is that the test loop is only called when :meth:`~lightning.pytorch.trainer.trainer.Trainer.test` is used.

.. code-block:: python

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    dataloader = DataLoader(dataset)
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=dataloader)

    # automatically loads the best weights for you
    trainer.test(model)

There are two ways to call ``test()``:

.. code-block:: python

    # call after training
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=dataloader)

    # automatically auto-loads the best weights from the previous run
    trainer.test(dataloaders=test_dataloaders)

    # or call with pretrained model
    model = LightningTransformer.load_from_checkpoint(PATH)
    dataset = WikiText2()
    test_dataloader = DataLoader(dataset)
    trainer = L.Trainer()
    trainer.test(model, dataloaders=test_dataloader)

.. note::
    `WikiText2` is used in a manner that does not create a train, test, val split. This is done for illustrative purposes only.
    A proper split can be created in :meth:`lightning.pytorch.core.LightningModule.setup` or :meth:`lightning.pytorch.core.LightningDataModule.setup`.

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

By default, the :meth:`~lightning.pytorch.core.LightningModule.predict_step` method runs the
:meth:`~lightning.pytorch.core.LightningModule.forward` method. In order to customize this behaviour,
simply override the :meth:`~lightning.pytorch.core.LightningModule.predict_step` method.

For the example let's override ``predict_step``:

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)

        def predict_step(self, batch):
            inputs, target = batch
            return self.model(inputs, target)

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
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=dataloader)

    # automatically auto-loads the best weights from the previous run
    predictions = trainer.predict(dataloaders=predict_dataloader)

    # or call with pretrained model
    model = LightningTransformer.load_from_checkpoint(PATH)
    dataset = WikiText2()
    test_dataloader = DataLoader(dataset)
    trainer = L.Trainer()
    predictions = trainer.predict(model, dataloaders=test_dataloader)

Inference in Research
=====================

If you want to perform inference with the system, you can add a ``forward`` method to the LightningModule.

.. note:: When using forward, you are responsible to call :func:`~torch.nn.Module.eval` and use the :func:`~torch.no_grad` context manager.

.. code-block:: python

    class LightningTransformer(L.LightningModule):
        def __init__(self, vocab_size):
            super().__init__()
            self.model = Transformer(vocab_size=vocab_size)

        def forward(self, batch):
            inputs, target = batch
            return self.model(inputs, target)

        def training_step(self, batch, batch_idx):
            inputs, target = batch
            output = self.model(inputs, target)
            loss = torch.nn.functional.nll_loss(output, target.view(-1))
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.1)


    model = LightningTransformer(vocab_size=dataset.vocab_size)

    model.eval()
    with torch.no_grad():
        batch = dataloader.dataset[0]
        pred = model(batch)

The advantage of adding a forward is that in complex systems, you can do a much more involved inference procedure,
such as text generation:

.. code-block:: python

    class Seq2Seq(L.LightningModule):
        def forward(self, x):
            embeddings = self(x)
            hidden_states = self.encoder(embeddings)
            for h in hidden_states:
                # decode
                ...
            return decoded

In the case where you want to scale your inference, you should be using
:meth:`~lightning.pytorch.core.LightningModule.predict_step`.

.. code-block:: python

    class Autoencoder(L.LightningModule):
        def forward(self, x):
            return self.decoder(x)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            # this calls forward
            return self(batch)


    data_module = ...
    model = Autoencoder()
    trainer = Trainer(accelerator="gpu", devices=2)
    trainer.predict(model, data_module)

Inference in Production
=======================

For cases like production, you might want to iterate different models inside a LightningModule.

.. code-block:: python

    from torchmetrics.functional import accuracy


    class ClassificationTask(L.LightningModule):
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

        trainer = Trainer(accelerator="gpu", devices=2)
        trainer.fit(task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

Tasks can be arbitrarily complex such as implementing GAN training, self-supervised or even RL.

.. code-block:: python

    class GANTask(L.LightningModule):
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
    trainer = Trainer(accelerator="gpu", devices=2)
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


********************
Save Hyperparameters
********************

Often times we train many versions of a model. You might share that model or come back to it a few months later at which
point it is very useful to know how that model was trained (i.e.: what learning rate, neural network, etc...).

Lightning has a standardized way of saving the information for you in checkpoints and YAML files. The goal here is to
improve readability and reproducibility.

save_hyperparameters
====================

Use :meth:`~lightning.pytorch.core.LightningModule.save_hyperparameters` within your
:class:`~lightning.pytorch.core.LightningModule`'s ``__init__`` method. It will enable Lightning to store all the
provided arguments under the ``self.hparams`` attribute. These hyperparameters will also be stored within the model
checkpoint, which simplifies model re-instantiation after training.

.. code-block:: python

    class LitMNIST(L.LightningModule):
        def __init__(self, layer_1_dim=128, learning_rate=1e-2):
            super().__init__()
            # call this to save (layer_1_dim=128, learning_rate=1e-4) to the checkpoint
            self.save_hyperparameters()

            # equivalent
            self.save_hyperparameters("layer_1_dim", "learning_rate")

            # Now possible to access layer_1_dim from hparams
            self.hparams.layer_1_dim


In addition, loggers that support it will automatically log the contents of ``self.hparams``.

Excluding hyperparameters
=========================

By default, every parameter of the ``__init__`` method will be considered a hyperparameter to the LightningModule.
However, sometimes some parameters need to be excluded from saving, for example when they are not serializable. Those
parameters should be provided back when reloading the LightningModule. In this case, exclude them explicitly:

.. code-block:: python

    class LitMNIST(L.LightningModule):
        def __init__(self, loss_fx, generator_network, layer_1_dim=128):
            super().__init__()
            self.layer_1_dim = layer_1_dim
            self.loss_fx = loss_fx

            # call this to save only (layer_1_dim=128) to the checkpoint
            self.save_hyperparameters("layer_1_dim")

            # equivalent
            self.save_hyperparameters(ignore=["loss_fx", "generator_network"])


load_from_checkpoint
====================

LightningModules that have hyperparameters automatically saved with
:meth:`~lightning.pytorch.core.LightningModule.save_hyperparameters` can conveniently be loaded and instantiated
directly from a checkpoint with :meth:`~lightning.pytorch.core.LightningModule.load_from_checkpoint`:

.. code-block:: python

    # to load specify the other args
    model = LitMNIST.load_from_checkpoint(PATH, loss_fx=torch.nn.SomeOtherLoss, generator_network=MyGenerator())


If parameters were excluded, they need to be provided at the time of loading:

.. code-block:: python

    # the excluded parameters were `loss_fx` and `generator_network`
    model = LitMNIST.load_from_checkpoint(PATH, loss_fx=torch.nn.SomeOtherLoss, generator_network=MyGenerator())


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

.. automethod:: lightning.pytorch.core.module.LightningModule.all_gather
    :noindex:

configure_callbacks
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.configure_callbacks
    :noindex:

configure_optimizers
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.configure_optimizers
    :noindex:

forward
~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.forward
    :noindex:

freeze
~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.freeze
    :noindex:

.. _lm-log:

log
~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.log
    :noindex:

log_dict
~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.log_dict
    :noindex:

lr_schedulers
~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.lr_schedulers
    :noindex:

manual_backward
~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.manual_backward
    :noindex:

optimizers
~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.optimizers
    :noindex:

print
~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.print
    :noindex:

predict_step
~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.predict_step
    :noindex:

save_hyperparameters
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.save_hyperparameters
    :noindex:

toggle_optimizer
~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.toggle_optimizer
    :noindex:

test_step
~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.test_step
    :noindex:

to_onnx
~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.to_onnx
    :noindex:

to_torchscript
~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.to_torchscript
    :noindex:

training_step
~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.training_step
    :noindex:

unfreeze
~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.unfreeze
    :noindex:

untoggle_optimizer
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.untoggle_optimizer
    :noindex:

validation_step
~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.validation_step
    :noindex:

-----------

Properties
==========

These are properties available in a LightningModule.

current_epoch
~~~~~~~~~~~~~

The number of epochs run.

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
            ...

global_step
~~~~~~~~~~~

The number of optimizer steps taken (does not reset each epoch).
This includes multiple optimizers (if enabled).

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.logger.experiment.log_image(..., step=self.global_step)

hparams
~~~~~~~

The arguments passed through ``LightningModule.__init__()`` and saved by calling
:meth:`~lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin.save_hyperparameters` could be accessed by the ``hparams`` attribute.

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
        # List of Logger objects
        loggers = self.loggers
        for logger in loggers:
            logger.log_metrics({"foo": 1.0})

local_rank
~~~~~~~~~~~

The ``local_rank`` is the index of the current process across all the devices for the current node.
You usually do not need to use this property, but it is useful to know how to access it if needed.
For example, if using 10 machines (or nodes), the GPU at index 0 on each machine has local_rank = 0.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.local_rank == 0:
            # do something only once across each node
            ...

precision
~~~~~~~~~

The type of precision used:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.precision == "16-true":
            ...

trainer
~~~~~~~

Pointer to the trainer

.. code-block:: python

    def training_step(self, batch, batch_idx):
        max_steps = self.trainer.max_steps
        any_flag = self.trainer.any_flag

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

See :ref:`manual optimization <common/optimization:Manual optimization>` for details.

.. code-block:: python

    def __init__(self):
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)

        loss = ...
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

Manual optimization is most useful for research topics like reinforcement learning, sparse coding, and GAN research.
It is required when you are using 2+ optimizers because with automatic optimization, you can only use one optimizer.

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

--------------

.. _lightning_hooks:

Hooks
=====

This is the pseudocode to describe the structure of :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`.
The inputs and outputs of each function are not represented for simplicity. Please check each function's API reference
for more information.

.. code-block:: python

    # runs on every device: devices can be GPUs, TPUs, ...
    def fit(self):
        configure_callbacks()

        if local_rank == 0:
            prepare_data()

        setup("fit")
        configure_model()
        configure_optimizers()

        on_fit_start()

        # the sanity check runs here

        on_train_start()
        for epoch in epochs:
            fit_loop()
        on_train_end()

        on_fit_end()
        teardown("fit")


    def fit_loop():
        torch.set_grad_enabled(True)

        on_train_epoch_start()

        for batch in train_dataloader():
            on_train_batch_start()

            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()

            out = training_step()

            on_before_zero_grad()
            optimizer_zero_grad()

            on_before_backward()
            backward()
            on_after_backward()

            on_before_optimizer_step()
            configure_gradient_clipping()
            optimizer_step()

            on_train_batch_end(out, batch, batch_idx)

            if should_check_val:
                val_loop()

        on_train_epoch_end()


    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)

        on_validation_start()
        on_validation_epoch_start()

        for batch_idx, batch in enumerate(val_dataloader()):
            on_validation_batch_start(batch, batch_idx)

            batch = on_before_batch_transfer(batch)
            batch = transfer_batch_to_device(batch)
            batch = on_after_batch_transfer(batch)

            out = validation_step(batch, batch_idx)

            on_validation_batch_end(out, batch, batch_idx)

        on_validation_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)

backward
~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.backward
    :noindex:

on_before_backward
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_before_backward
    :noindex:

on_after_backward
~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_after_backward
    :noindex:

on_before_zero_grad
~~~~~~~~~~~~~~~~~~~
.. automethod:: lightning.pytorch.core.module.LightningModule.on_before_zero_grad
    :noindex:

on_fit_start
~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_fit_start
    :noindex:

on_fit_end
~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_fit_end
    :noindex:


on_load_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_load_checkpoint
    :noindex:

on_save_checkpoint
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_save_checkpoint
    :noindex:

load_from_checkpoint
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.load_from_checkpoint
    :noindex:

on_train_start
~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_start
    :noindex:

on_train_end
~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_end
    :noindex:

on_validation_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_start
    :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_end
    :noindex:

on_test_batch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_batch_start
    :noindex:

on_test_batch_end
~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_batch_end
    :noindex:

on_test_epoch_start
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_epoch_start
    :noindex:

on_test_epoch_end
~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_epoch_end
    :noindex:

on_test_start
~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_start
    :noindex:

on_test_end
~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_end
    :noindex:

on_predict_batch_start
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_batch_start
    :noindex:

on_predict_batch_end
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_batch_end
    :noindex:

on_predict_epoch_start
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_epoch_start
    :noindex:

on_predict_epoch_end
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_epoch_end
    :noindex:

on_predict_start
~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_start
    :noindex:

on_predict_end
~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_predict_end
    :noindex:

on_train_batch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_batch_start
    :noindex:

on_train_batch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_batch_end
    :noindex:

on_train_epoch_start
~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_epoch_start
    :noindex:

on_train_epoch_end
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_train_epoch_end
    :noindex:

on_validation_batch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_batch_start
    :noindex:

on_validation_batch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_batch_end
    :noindex:

on_validation_epoch_start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_epoch_end
    :noindex:

configure_model
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.configure_model
    :noindex:

on_validation_model_eval
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_model_eval
    :noindex:

on_validation_model_train
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_validation_model_train
    :noindex:

on_test_model_eval
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_model_eval
    :noindex:

on_test_model_train
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_test_model_train
    :noindex:

on_before_optimizer_step
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_before_optimizer_step
    :noindex:

configure_gradient_clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.configure_gradient_clipping
    :noindex:

optimizer_step
~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.optimizer_step
    :noindex:

optimizer_zero_grad
~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.optimizer_zero_grad
    :noindex:

prepare_data
~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.prepare_data
    :noindex:

setup
~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.setup
    :noindex:

teardown
~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.teardown
    :noindex:

train_dataloader
~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.train_dataloader
    :noindex:

val_dataloader
~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.val_dataloader
    :noindex:

test_dataloader
~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.test_dataloader
    :noindex:

predict_dataloader
~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.predict_dataloader
    :noindex:

transfer_batch_to_device
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.transfer_batch_to_device
    :noindex:

on_before_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_before_batch_transfer
    :noindex:

on_after_batch_transfer
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lightning.pytorch.core.module.LightningModule.on_after_batch_transfer
    :noindex:
