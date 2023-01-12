:orphan:

#########
Callbacks
#########

Callbacks enable you, or the users of your code, to add new behavior to the training loop without the need to modify the source code in-place.


----------


*************************************
Add a callback interface to your loop
*************************************

Suppose we want to enable anyone to run some arbitrary code at the end of a training iteration.
Here is how that gets done in Fabric:

.. code-block:: python
    :caption: my_callbacks.py

    class MyCallback:
        def on_train_batch_end(self, loss, output):
            # Here, put yny code you want to run at the end of a training step
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
        # Give the callback access to some varibles
        fabric.call("on_train_batch_end", loss=loss, output=...)


As you can see, the code inside the callback method is completely decoupled from the trainer code.
This enables flexibility in extending the loop in arbitrary ways.

**Exercise**: Implement a callback that computes and prints the time to complete an iteration.


----------


******************
Multiple callbacks
******************

The callback system is designed so that it can run multiple callbacks at the same time.
You can simply pass a list to Fabric:

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


The :meth:`~lightning_fabric.fabric.Fabric.call` simply takes care of calling the callback objects in the order they are given to Fabric.
Not all objects registered via ``Fabric(callbacks=...)`` must implement a method with the given name.
The ones that have a matching method name will get called.


**********
Next steps
**********

Callbacks are a powerful tool to build a Trainer. Learn how in our comprehensive guide.

