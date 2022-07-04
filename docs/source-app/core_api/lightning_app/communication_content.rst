

**************************************************
What communication between components does for you
**************************************************

When creating interactive apps with multiple components, you might want your components to share information with each other. You might to rely on that information to control their execution, share progress in the UI, trigger a sequence of operations, etc.

By design, the :class:`~lightning_app.core.flow.LightningFlow` communicates to all :class:`~lightning_app.core.flow.LightningWork` within the application, but :class:`~lightning_app.core.flow.LightningWork` can't communicate between each other directly, they need the flow as a proxy to do so.

Once a ``LightningWork`` is running, any updates to its state is automatically communicated to the flow as a delta (using `DeepDiff <https://github.com/seperman/deepdiff>`_). The state communication isn't bi-directional, it is only done from work to flow.

Internally, the Lightning App is alternatively collecting deltas sent from all the registered ``LightningWorks`` and/or UI, and running the root flow run method of the app.

----

*************************************************
Communication from LightningWork to LightningFlow
*************************************************

Here's an example to better understand communication from LightningWork to LightningFlow.

The ``WorkCounter`` increments a counter until 1 million and the ``Flow`` prints the work counter.

As the LightningWork is running into its own process, its state changes is sent to the LightningFlow which contains the latest value of the counter.

.. code-block:: python

    import lightning as L


    class WorkCounter(L.LightningWork):
        def __init__(self):
            super().__init__(parallel=True)
            self.counter = 0

        def run(self):
            for _ in range(int(10e6)):
                self.counter += 1


    class Flow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.w = WorkCounter()

        def run(self):
            self.w.run()
            print(self.w.counter)


    app = L.LightningApp(Flow())


A delta sent from the LightningWork to the LightningFlow looks like this:

.. code-block:: python

    {"values_changed": {"root['works']['w']['vars']['counter']": {"new_value": 425}}}

Here is the associated illustration:

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/deltas.gif
    :alt: Mechanism showing how delta are sent.
    :width: 100 %

----

*************************************************
Communication from LightningFlow to LightningWork
*************************************************

Communication from the LightningFlow to the LightningWork while running **isn't support yet**. If your application requires this feature, please open an issue on Github.

.. code-block:: python

    import lightning as L
    from time import sleep


    class WorkCounter(L.LightningWork):
        def __init__(self):
            super().__init__(parallel=True)
            self.counter = 0

        def run(self):
            while True:
                sleep(1)
                print(f"Work {self.counter}")


    class Flow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.w = WorkCounter()

        def run(self):
            self.w.run()
            sleep(1)
            print(f"Flow {self.w.counter}")
            self.w.counter += 1


    app = L.LightningApp(Flow())

As you can observe, there is a divergence between the value within the LightningWork and the LightningFlow.

.. code-block:: console

    Flow 0
    Flow 1
    Flow 2
    Flow 3
    Work 0
    Flow 4
    Work 0
    Flow 5
    Work 0
    Flow 6
    Work 0
    Flow 7
    Work 0
    Flow 8
    Work 0
    Flow 9
    Work 0
    Flow 10
