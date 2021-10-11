.. _loop_customization:

Loops
=====

The Lightning Trainer automates the optimization loop to save you from writing boilerplate.
Need more control over the optimization scheme to try different training paradigms such as recommendation engine optimization or active learning?
You can now customize the built-in Lightning training loop, providing an unprecedented level of flexibility.

With customized loops, you can not only customize Lightning down to its very core, but also build new functionalities on top of it.
Until now, Lightning did not support well some training paradigms like recommendation engine optimization or active learning.
The loop customization feature will not only enable researchers to customize Lightning down to its very core, but also allow one to build new functionalities on top of it.

.. image:: ../_static/images/extensions/loops/epoch-loop-steps.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop


The Built-in Training Loop
--------------------------

Every PyTorch user is familiar with the basic training loop for gradient descent optimization:

.. code-block:: python

    for i, batch in enumerate(dataloader):
        x, y = batch
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

At its core, the Lightning Trainer does not do anything different here.
It implements the same loop as shown above except that the research code stays in the LightningModule:

.. code-block:: python

    for i, batch in enumerate(dataloader):
        loss = lightning_module.training_step(batch, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

The other operations are automated by the Lightning Trainer: :code:`zero_grad()`, :code:`backward()` and :code:`optimizer.step()` calls.
These are considered *boilerplate* and get automated by Lightning.

This optimization scheme is very general and applies to the vast majority of deep learning research today.
However, the loops and optimizer calls here remain predetermined in their order and are fully controlled by the Trainer.

Custom Lightning Loops
----------------------

With the new :class:`~pytorch_lightning.loops.base.Loop` class, you can now fully customize the optimization scheme for any use case.

Here is how the above training loop can be defined using the new Loop API:

.. code-block:: python

    class EpochLoop(Loop):
        def advance(self, batch, i):
            loss = lightning_module.training_step(batch, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def run(self, dataloader):
            for i, batch in enumerate(dataloader):
                self.advance(batch, i)

Defining a loop with a class interface instead of hard-coding a raw Python for/while loop has several benefits:

1. You can have full control over the data flow through loops
2. You can add new loops and nest as many of them as you want
3. If needed, the state of a loop can be saved and resumed
4. New hooks can be injected at any point

Which loops does Lightning have and how can they be changed?
------------------------------------------------------------

.. image:: ../_static/images/extensions/loops/replace-fit-loop.gif
    :alt: Animation showing how to replace a loop on the Trainer

Entry Loops: Fit, Validate, Test, Predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Trainer has four entry points for training, testing and inference, and each method corresponds to a main loop:

.. list-table:: Trainer entry points and associated loops
   :widths: 25 25 25
   :header-rows: 1

   * - Entry point
     - Trainer attribute
     - Loop class
   * - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
     - :attr:`~pytorch_lightning.trainer.trainer.Trainer.fit_loop`
     - :class:`~pytorch_lightning.loops.fit_loop.FitLoop`
   * - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
     - :attr:`~pytorch_lightning.trainer.trainer.Trainer.validate_loop`
     - :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop`
   * - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
     - :attr:`~pytorch_lightning.trainer.trainer.Trainer.test_loop`
     - :class:`~pytorch_lightning.loops.dataloader.evaluation_loop.EvaluationLoop`
   * - :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`
     - :attr:`~pytorch_lightning.trainer.trainer.Trainer.predict_loop`
     - :class:`~pytorch_lightning.loops.dataloader.prediction_loop.PredictionLoop`


When the user calls :code:`Trainer.<entry-point>`, it redirects to the corresponding :code:`Trainer.loop.run()` which implements the main logic of that particular Lightning loop.
The :meth:`~pytorch_lightning.loops.base.Loop.run` method is part of the base :class:`~pytorch_lightning.loops.base.Loop` class that every loop inherits from (like every model inherits from LightningModule).

Customizing any of these entry point loops is simple:

**Step 1:** Subclass one of the above loop classes (or inherit `Loop` to start from scratch)

.. code-block:: python

    from pytorch_lightning.loops import FitLoop


    class MyLoop(FitLoop):
        ...

Alternatively, more advanced users can also implement a complete loop from scratch by inheriting directly from the base loop interface as explained later.

**Step 2:** Attach the loop to the Trainer and run it.

.. code-block:: python

    loop = MyLoop()
    trainer = Trainer()

    # replace the loop directly on the Trainer
    trainer.fit_loop = loop
    trainer.fit(model)

    # or
    trainer.validate_loop = loop
    trainer.validate(model)

    # or
    trainer.test_loop = loop
    trainer.test(model)

    # or
    trainer.predict_loop = loop
    trainer.predict(model)


Subloops
^^^^^^^^

.. image:: ../_static/images/extensions/loops/connect-epoch-loop.gif
    :alt: Animation showing how to replace a subloop of `Trainer.fit()`

When loops have subloops (nested loops), they can be changed and switched out directly with the :meth:`~pytorch_lightning.loops.base.Loop.connect` method.
An example is the built-in :class:`~pytorch_lightning.loops.fit_loop.FitLoop`, which has the EpochLoop as its subloop.
Customizing the EpochLoop does not require you to implement an entirely new fit loop:

.. code-block:: python

    # Step 1: create your loop
    my_epoch_loop = MyEpochLoop()

    # Step 2: use connect()
    trainer.fit_loop.connect(epoch_loop=my_epoch_loop)

    # Trainer runs the fit loop with your new epoch loop!
    trainer.fit(model)

More about the built-in loops and how they are composed is explained :ref:`here <loop structure>`:

The Loop Base Class
-------------------

So far we have seen how it is possible to customize existing implementations of loops in Lightning, namely the FitLoop and the OptimizerLoop.
This is an appropriate approach when just a few details need change.
But when a loop needs to perform a fundamentally different function, it is better to implement the entire loop by inheriting from the base :class:`~pytorch_lightning.loops.base.Loop` interface.

The :class:`~pytorch_lightning.loops.base.Loop` class is the base for all loops in Lighting just like the LightningModule is the base for all models.
It defines a public interface that each loop implementation must follow, the key ones are:

- :meth:`~pytorch_lightning.loops.base.Loop.advance`: implements the logic of a single iteration in the loop
- :meth:`~pytorch_lightning.loops.base.Loop.done`: a boolean stopping criteria
- :meth:`~pytorch_lightning.loops.base.Loop.reset`: implements a mechanism to reset the loop so it can be restarted

These methods are called by the default implementation of the :meth:`~pytorch_lightning.loops.base.Loop.run` entry point as shown in the (reduced) code excerpt below.

.. code-block:: python

    def run(self, *args, **kwargs):

        self.reset()
        self.on_run_start(*args, **kwargs)

        while not self.done:
            try:
                self.advance(*args, **kwargs)
            except StopIteration:
                break

        output = self.on_run_end()
        return output

Some important observations here: One, the ``run()`` method can define input arguments that get forwarded to some of the other methods that get invoked as part of ``run()``.
Such input arguments typically comprise of one or several iterables over which the loop is supposed to iterate, for example, an iterator over a :class:`~torch.utils.data.DataLoader`.
The reason why the inputs get forwarded is mainly for convenience but implementations are free to change this.
Secondly, ``advance()`` can raise a :class:`StopIteration` to exit the loop early.
This is analogous to a :code:`break` statement in a raw Python ``while`` for example.
Finally, a loop may return an output as part of ``run()``.
As an example, the loop could return a list containing all results produced in each iteration (advance).

Loops can also be nested! That is, a loop may call another one inside of its ``advance()``.


Example: Adding your own hooks
------------------------------

The loop manage all calls to hooks in LightningModule and Callbacks.
With loop customization you have the ability to change these calls or add new ones.
Simply subclass one of the :ref:`built-in loops <loop structure>` and add the call at the place you need it:

.. code-block:: python

    from pytorch_lightning.loops import TrainingEpochLoop


    class CustomEpochLoop(TrainingEpochLoop):
        def advance(self):
            ...
            self.trainer.lightning_module.my_new_hook(*args, **kwargs)
            ...


More about the built-in loops and how they are composed is explained :ref:`here <loop structure>`:


Showcase: Active Learning Loop in Lightning Flash
-------------------------------------------------

`Lightning Flash <https://github.com/PyTorchLightning/lightning-flash>`__ is already using custom loops to implement new tasks!
`Active Learning <https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`__ is a machine learning practice in which the user interacts with the learner in order to provide new labels when required.
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
        val_split=0.1,
    )

    # 2. Build the task
    head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(512, datamodule.num_classes),
    )
    model = ImageClassifier(backbone="resnet18", head=head, num_classes=datamodule.num_classes, serializer=Probabilities())


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

Here is the `runnable example <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash_examples/integrations/baal/image_classification_active_learning.py>`_ and the `code for the active learning loop <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/image/classification/integrations/baal/loop.py#L31>`_.


.. _loop structure:

Built-in Loop Structure
-----------------------

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


Each of these :code:`for`-loops represents a class implementing the :class:`~pytorch_lightning.loops.base.Loop` interface.


.. list-table:: Trainer entry points and associated loops
   :widths: 25 75
   :header-rows: 1

   * - Built-in loop
     - Description
   * - :class:`~pytorch_lightning.loops.fit_loop.FitLoop`
     - The :class:`~pytorch_lightning.loops.fit_loop.FitLoop` is the top-level loop where training starts.
       It simply counts the epochs and iterates from one to the next by calling :code:`TrainingEpochLoop.run()` in its :code:`advance()` method.
   * - :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop`
     - The :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` is the one that iterates over the dataloader that the user returns in their :meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader` method.
       Its main responsibilities are calling the :code:`*_epoch_start` and :code:`*_epoch_end` hooks, accumulating outputs if the user request them in one of these hooks, and running validation at the requested interval.
       The validation is carried out by yet another loop, :class:`~pytorch_lightning.loops.epoch.validation_epoch_loop.ValidationEpochLoop`.

       In the :code:`run()` method, the training epoch loop could in theory simply call the :code:`LightningModule.training_step` already and perform the optimization.
       However, Lightning has built-in support for automatic optimization with multiple optimizers and on top of that also supports :doc:`truncated back-propagation through time <../advanced/sequences>`.
       For this reason there are actually two more loops nested under :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop`.
   * - :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop`
     - The responsibility of the :class:`~pytorch_lightning.loops.batch.training_batch_loop.TrainingBatchLoop` is to split a batch given by the :class:`~pytorch_lightning.loops.epoch.training_epoch_loop.TrainingEpochLoop` along the time-dimension and iterate over the list of splits.
       It also keeps track of the hidden state *hiddens* returned by the training step.
       By default, when truncated back-propagation through time (TBPTT) is turned off, this loop does not do anything except redirect the call to the :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop`.
       Read more about :doc:`TBPTT <../advanced/sequences>`.
   * - :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop`
     - The :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` iterates over one or multiple optimizers and for each one it calls the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` method with the batch, the current batch index and the optimizer index if multiple optimizers are requested.
       It is the leaf node in the tree of loops and performs the actual optimization (forward, zero grad, backward, optimizer step).
   * - :class:`~pytorch_lightning.loops.optimization.manual_loop.ManualOptimization`
     - Substitutes the :class:`~pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop` in case of :ref:`manual_optimization` and implements the manual optimization step.


Advanced Topics and Examples
----------------------------

Next: :doc:`Advanced loop features and examples <../extensions/loops_advanced>`
