.. _loop-customization-extensions:


Loops
=====

Loops let advanced users swap out the default gradient descent optimization loop at the core of Lightning with a different optimization paradigm.

The Lightning Trainer is built on top of the standard gradient descent optimization loop which works for 90%+ of machine learning use cases:

.. code-block:: python

    for i, batch in enumerate(dataloader):
        x, y = batch
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

However, some new research use cases such as meta-learning, active learning, recommendation systems, etc., require a different loop structure.
For example here is a simple loop that guides the weight updates with a loss from a special validation split:

.. code-block:: python

    for i, batch in enumerate(train_dataloader):
        x, y = batch
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        optimizer.zero_grad()
        loss.backward()

        val_loss = 0
        for i, val_batch in enumerate(val_dataloader):
            x, y = val_batch
            y_hat = model(x)
            val_loss += loss_function(y_hat, y)

        scale_gradients(model, 1 / val_loss)
        optimizer.step()


With Lightning Loops, you can customize to non-standard gradient descent optimizations to get the same loop above:

.. code-block:: python

    trainer = Trainer()
    trainer.fit_loop.epoch_loop = MyGradientDescentLoop()

Think of this as swapping out the engine in a car!

----------

Understanding the Default Trainer Loop
--------------------------------------

The Lightning :class:`~pytorch_lightning.trainer.trainer.Trainer` automates the standard optimization loop which every PyTorch user is familiar with:

.. code-block:: python

    for i, batch in enumerate(dataloader):
        x, y = batch
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

The core research logic is simply shifted to the :class:`~pytorch_lightning.core.module.LightningModule`:

.. code-block:: python

    for i, batch in enumerate(dataloader):
        # x, y = batch                      moved to training_step
        # y_hat = model(x)                  moved to training_step
        # loss = loss_function(y_hat, y)    moved to training_step
        loss = lightning_module.training_step(batch, i)

        # Lightning handles automatically:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Under the hood, the above loop is implemented using the :class:`~pytorch_lightning.loops.loop.Loop` API like so:

.. code-block:: python

    class DefaultLoop(Loop):
        def advance(self, batch, i):
            loss = lightning_module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def run(self, dataloader):
            for i, batch in enumerate(dataloader):
                self.advance(batch, i)

Defining a loop within a class interface instead of hard-coding a raw Python for/while loop has several benefits:

1. You can have full control over the data flow through loops.
2. You can add new loops and nest as many of them as you want.
3. If needed, the state of a loop can be :ref:`saved and resumed <persisting loop state>`.
4. New hooks can be injected at any point.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/loops/epoch-loop-steps.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop

----------

.. _override-default-loops-extensions:

Overriding the default Loops
----------------------------

The fastest way to get started with loops, is to override functionality of an existing loop.
Lightning has 4 main loops which relies on : :class:`~pytorch_lightning.loops.fit_loop.FitLoop` for fitting (training and validating),
:class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop` for validating or testing,
:class:`~pytorch_lightning.loops.dataloader.prediction_loop.PredictionLoop` for predicting.

For simple changes that don't require a custom loop, you can modify each of these loops.

Each loop has a series of methods that can be modified.
For example with the :class:`~pytorch_lightning.loops.fit_loop.FitLoop`:

.. code-block:: python

    from pytorch_lightning.loops import FitLoop


    class MyLoop(FitLoop):
        def advance(self):
            """Advance from one iteration to the next."""

        def on_advance_end(self):
            """Do something at the end of an iteration."""

        def on_run_end(self):
            """Do something when the loop ends."""

A full list with all built-in loops and subloops can be found :ref:`here <loop-structure-extensions>`.

To add your own modifications to a loop, simply subclass an existing loop class and override what you need.
Here is a simple example how to add a new hook:

.. code-block:: python

    from pytorch_lightning.loops import FitLoop


    class CustomFitLoop(FitLoop):
        def advance(self):
            # ... whatever code before

            # pass anything you want to the hook
            self.trainer.call_hook("my_new_hook", *args, **kwargs)

            # ... whatever code after

Now simply attach the correct loop in the trainer directly:

.. code-block:: python

    trainer = Trainer(...)
    trainer.fit_loop = CustomFitLoop()

    # fit() now uses the new FitLoop!
    trainer.fit(...)

    # the equivalent for validate()
    val_loop = CustomValLoop()
    trainer = Trainer()
    trainer.validate_loop = val_loop
    trainer.validate(...)

Now your code is FULLY flexible and you can still leverage ALL the best parts of Lightning!

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/loops/replace-fit-loop.gif
    :alt: Animation showing how to replace a loop on the Trainer

----------

Creating a New Loop From Scratch
--------------------------------

You can also go wild and implement a full loop from scratch by sub-classing the :class:`~pytorch_lightning.loops.loop.Loop` base class.
You will need to override a minimum of two things:

.. code-block:: python

    from pytorch_lightning.loop import Loop


    class MyFancyLoop(Loop):
        @property
        def done(self):
            """Provide a condition to stop the loop."""

        def advance(self):
            """
            Access your dataloader/s in whatever way you want.
            Do your fancy optimization things.
            Call the LightningModule methods at your leisure.
            """

Finally, attach it into the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. code-block:: python

    trainer = Trainer(...)
    trainer.fit_loop = MyFancyLoop()

    # fit() now uses your fancy loop!
    trainer.fit(...)

But beware: Loop customization gives you more power and full control over the Trainer and with great power comes great responsibility.
We recommend that you familiarize yourself with :ref:`overriding the default loops <override-default-loops-extensions>` first before you start building a new loop from the ground up.

----------

Loop API
--------
Here is the full API of methods available in the Loop base class.

The :class:`~pytorch_lightning.loops.loop.Loop` class is the base of all loops in the same way as the :class:`~pytorch_lightning.core.module.LightningModule` is the base of all models.
It defines a public interface that each loop implementation must follow, the key ones are:

Properties
^^^^^^^^^^

done
~~~~

.. autoattribute:: pytorch_lightning.loops.loop.Loop.done
    :noindex:

skip (optional)
~~~~~~~~~~~~~~~

.. autoattribute:: pytorch_lightning.loops.loop.Loop.skip
    :noindex:

Methods
^^^^^^^

reset (optional)
~~~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.loops.loop.Loop.reset
    :noindex:

advance
~~~~~~~

.. automethod:: pytorch_lightning.loops.loop.Loop.advance
    :noindex:

run (optional)
~~~~~~~~~~~~~~

.. automethod:: pytorch_lightning.loops.loop.Loop.run
    :noindex:


----------

Subloops
--------

When you want to customize nested loops within loops, use the :meth:`~pytorch_lightning.loops.loop.Loop.replace` method:

.. code-block:: python

    # This takes care of properly instantiating the new Loop and setting all references
    trainer.fit_loop.replace(epoch_loop=MyEpochLoop)
    # Trainer runs the fit loop with your new epoch loop!
    trainer.fit(model)

Alternatively, for more fine-grained control, use the :meth:`~pytorch_lightning.loops.loop.Loop.connect` method:

.. code-block:: python

    # Optional: stitch back the trainer arguments
    epoch_loop = MyEpochLoop(trainer.fit_loop.epoch_loop.min_steps, trainer.fit_loop.epoch_loop.max_steps)
    # Optional: connect children loops as they might have existing state
    epoch_loop.connect(trainer.fit_loop.epoch_loop.batch_loop, trainer.fit_loop.epoch_loop.val_loop)
    # Instantiate and connect the loop.
    trainer.fit_loop.connect(epoch_loop=epoch_loop)
    trainer.fit(model)

More about the built-in loops and how they are composed is explained in the next section.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/loops/connect-epoch-loop.gif
    :alt: Animation showing how to connect a custom subloop

----------

Built-in Loops
--------------

.. _loop-structure-extensions:

The training loop in Lightning is called *fit loop* and is actually a combination of several loops.
Here is what the structure would look like in plain Python:

.. code-block:: python

    # FitLoop
    for epoch in range(max_epochs):

        # TrainingEpochLoop
        for batch_idx, batch in enumerate(train_dataloader):

            # TrainingBatchLoop
            for split_batch in tbptt_split(batch):

                # OptimizerLoop
                for optimizer_idx, opt in enumerate(optimizers):

                    loss = lightning_module.training_step(batch, batch_idx, optimizer_idx)
                    ...

            # ValidationEpochLoop
            for batch_idx, batch in enumerate(val_dataloader):
                lightning_module.validation_step(batch, batch_idx, optimizer_idx)
                ...


Each of these :code:`for`-loops represents a class implementing the :class:`~pytorch_lightning.loops.loop.Loop` interface.


.. list-table:: Trainer entry points and associated loops
   :widths: 25 75
   :header-rows: 1

   * - Built-in loop
     - Description
   * - :class:`~pytorch_lightning.loops.fit_loop.FitLoop`
     - The :class:`~pytorch_lightning.loops.fit_loop.FitLoop` is the top-level loop where training starts.
       It simply counts the epochs and iterates from one to the next by calling :code:`TrainingEpochLoop.run()` in its :code:`advance()` method.
   * - :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop`
     - The :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` is the one that iterates over the dataloader that the user returns in their :meth:`~pytorch_lightning.core.module.LightningModule.train_dataloader` method.
       Its main responsibilities are calling the :code:`*_epoch_start` and :code:`*_epoch_end` hooks, accumulating outputs if the user request them in one of these hooks, and running validation at the requested interval.
       The validation is carried out by yet another loop, :class:`~pytorch_lightning.loops.epoch.validation_epoch_loop.ValidationEpochLoop`.

       In the :code:`run()` method, the training epoch loop could in theory simply call the :code:`LightningModule.training_step` already and perform the optimization.
       However, Lightning has built-in support for automatic optimization with multiple optimizers and on top of that also supports :ref:`TBPTT <sequential-data>`.
       For this reason there are actually two more loops nested under :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop`.
   * - :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop`
     - The responsibility of the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop` is to split a batch given by the :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` along the time-dimension and iterate over the list of splits.
       It also keeps track of the hidden state *hiddens* returned by the training step.
       By default, when truncated back-propagation through time (TBPTT) is turned off, this loop does not do anything except redirect the call to the :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop`.
       Read more about :ref:`TBPTT <sequential-data>`.
   * - :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop`
     - The :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` iterates over one or multiple optimizers and for each one it calls the :meth:`~pytorch_lightning.core.module.LightningModule.training_step` method with the batch, the current batch index and the optimizer index if multiple optimizers are requested.
       It is the leaf node in the tree of loops and performs the actual optimization (forward, zero grad, backward, optimizer step).
   * - :class:`~pytorch_lightning.loops.optimization.manual_loop.ManualOptimization`
     - Substitutes the :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` in case of :doc:`manual optimization <../model/manual_optimization>` and implements the manual optimization step.
   * - :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop`
     - The :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop` is the top-level loop where validation/testing starts.
       It simply iterates over each evaluation dataloader from one to the next by calling :code:`EvaluationEpochLoop.run()` in its :code:`advance()` method.
   * - :class:`~pytorch_lightning.loops.dataloader.prediction_loop.PredictionLoop`
     - The :class:`~pytorch_lightning.loops.dataloader.prediction_loop.PredictionLoop` is the top-level loop where prediction starts.
       It simply iterates over each prediction dataloader from one to the next by calling :code:`PredictionEpochLoop.run()` in its :code:`advance()` method.


----------

Available Loops in Lightning Flash
----------------------------------

`Active Learning <https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`__ is a machine learning practice in which the user interacts with the learner in order to provide new labels when required.

You can find a real use case in `Lightning Flash <https://github.com/PyTorchLightning/lightning-flash>`_.

Flash implements the :code:`ActiveLearningLoop` that you can use together with the :code:`ActiveLearningDataModule` to label new data on the fly.
To run the following demo, install Flash and `BaaL <https://github.com/ElementAI/baal>`__  first:

.. code-block:: bash

    pip install lightning-flash baal

.. code-block:: python

    import torch

    import flash
    from flash.core.classification import Probabilities
    from flash.core.data.utils import download_data
    from flash.image import ImageClassificationData, ImageClassifier
    from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop

    # 1. Create the DataModule
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

    # Implement the research use-case where we mask labels from labelled dataset.
    datamodule = ActiveLearningDataModule(
        ImageClassificationData.from_folders(train_folder="data/hymenoptera_data/train/", batch_size=2),
        initial_num_labels=5,
        val_split=0.1,
    )

    # 2. Build the task
    head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(512, datamodule.num_classes),
    )
    model = ImageClassifier(backbone="resnet18", head=head, num_classes=datamodule.num_classes, output=Probabilities())


    # 3.1 Create the trainer
    trainer = flash.Trainer(max_epochs=3)

    # 3.2 Create the active learning loop and connect it to the trainer
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=1)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop

    # 3.3 Finetune
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 4. Predict what's on a few images! ants or bees?
    predictions = model.predict("data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg")
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")

Here is the `Active Learning Loop example <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash_examples/integrations/baal/image_classification_active_learning.py>`_ and the `code for the active learning loop <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/classification/integrations/baal/loop.py>`_.


----------

Advanced Examples
-----------------


.. list-table:: Ready-to-run loop examples and tutorials
   :widths: 25 75
   :header-rows: 1

   * - Link to Example
     - Description
   * - `K-fold Cross Validation <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/examples/pl_loops/kfold.py>`_
     - `KFold / Cross Validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`__ is a machine learning practice in which the training dataset is being partitioned into ``num_folds`` complementary subsets.
       One cross validation round will perform fitting where one fold is left out for validation and the other folds are used for training.
       To reduce variability, once all rounds are performed using the different folds, the trained models are ensembled and their predictions are
       averaged when estimating the model's predictive performance on the test dataset.
   * - `Yielding Training Step <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/examples/pl_loops/yielding_training_step.py>`_
     - This loop enables you to write the :meth:`~pytorch_lightning.core.module.LightningModule.training_step` hook
       as a Python Generator for automatic optimization with multiple optimizers, i.e., you can :code:`yield` loss
       values from it instead of returning them. This can enable more elegant and expressive implementations, as shown
       shown with a GAN in this example.


----------

Advanced Features
-----------------

Next: :doc:`Advanced loop features <../extensions/loops_advanced>`
