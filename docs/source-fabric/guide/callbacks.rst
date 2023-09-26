#########
Callbacks
#########

Callbacks enable you, or the users of your code, to add new behavior to the training loop without needing to modify the source code.


----


*************************************
Add a callback interface to your loop
*************************************

Suppose we want to enable anyone to run some arbitrary code at the end of a training iteration.
Here is how that gets done in Fabric:

.. code-block:: python
    :caption: my_callbacks.py

    class MyCallback:
        def on_train_batch_end(self, loss, output):
            # Here, put any code you want to run at the end of a training step
            ...


.. code-block:: python
    :caption: train.py
    :emphasize-lines: 4,7,18

    from lightning.fabric import Fabric

    # The code of a callback can live anywhere, away from the training loop
    from my_callbacks import MyCallback

    # Add one or several callbacks:
    fabric = Fabric(callbacks=[MyCallback()])

    ...

    for iteration, batch in enumerate(train_dataloader):
        ...
        fabric.backward(loss)
        optimizer.step()

        # Let a callback add some arbitrary processing at the appropriate place
        # Give the callback access to some variables
        fabric.call("on_train_batch_end", loss=loss, output=...)


As you can see, the code inside the callback method is completely decoupled from the trainer code.
This enables flexibility in extending the loop in arbitrary ways.

**Exercise**: Implement a callback that computes and prints the time to complete an iteration.


----


******************
Multiple callbacks
******************

The callback system is designed to easily run multiple callbacks at the same time.
You can pass a list to Fabric:

.. code-block:: python

    # Add multiple callback implementations in a list
    callback1 = LearningRateMonitor()
    callback2 = Profiler()
    fabric = Fabric(callbacks=[callback1, callback2])

    # Let Fabric call the implementations (if they exist)
    fabric.call("any_callback_method", arg1=..., arg2=...)

    # fabric.call is the same as doing this
    callback1.any_callback_method(arg1=..., arg2=...)
    callback2.any_callback_method(arg1=..., arg2=...)


The :meth:`~lightning.fabric.fabric.Fabric.call` calls the callback objects in the order they were given to Fabric.
Not all objects registered via ``Fabric(callbacks=...)`` must implement a method with the given name.
The ones that have a matching method name will get called.


----


**********
Next steps
**********

Callbacks are a powerful tool for building a Trainer.
See a real example of how they can be integrated in our Trainer template based on Fabric:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Trainer Template
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>
