
********************************
Communication Between Components
********************************

When creating interactive Lightning Apps (App) with multiple components, you may need your components to share information with each other and rely on that information to control their execution, share progress in the UI, trigger a sequence of operations, etc.

To accomplish that, Lightning components can communicate using the App State. The App State is composed of all attributes defined within each component's **__init__** method e.g anything attached to the component with **self.x = y**.

All attributes of all **LightningWork (Work)** components are accessible in the **LightningFlow (Flow)** components in real-time.

By design, the Flows communicate to all **Works** within the application. However, Works can't communicate with each other directly, they must use Flows as a proxy to communicate.

Once a Work is running, any updates to the Work's state is automatically communicated to the Flow, as a delta (using `DeepDiff <https://github.com/seperman/deepdiff>`_). The state communication isn't bi-directional, communication is only done from Work to Flow.

Internally, the App is alternatively collecting deltas sent from all the registered Works and/or UI, and running the root Flow run method of the App.

----

*************************************************
Communication from LightningWork to LightningFlow
*************************************************

LightningFlow (Flow) can access their children's LightningWork (Work) state.

When a running Work attribute gets updated inside its method (separate process locally or remote machine), the app re-executes Flow's run method once it receives the state update from the Work.

Here's an example to better understand communication from Work to Flow.

The ``WorkCounter`` increments a counter until 1 million and the ``Flow`` prints the work counter.

As the Work is running its own process, its state changes are sent to the Flow which contains the latest value of the counter.

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


A delta sent from the Work to the Flow looks like this:

.. code-block:: python

    {"values_changed": {"root['works']['w']['vars']['counter']": {"new_value": 425}}}

Here is the associated illustration:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/deltas.gif
    :alt: Mechanism showing how delta are sent.
    :width: 100 %

Here's another example that is slightly different. Here we define a Flow and Work, where the Work increments a counter indefinitely and the Flow prints its state which contain the Work.

You can easily check the state of your entire app as follows:

.. literalinclude:: ../../core_api/lightning_app/app.py

Run the app with:

.. code-block:: bash

    lightning run app docs/source/core_api/lightning_app/app.py

And here's the output you get when running the App using the **Lightning CLI**:

.. code-block:: console

    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    State: {'works': {'w': {'vars': {'counter': 1}}}}
    State: {'works': {'w': {'vars': {'counter': 2}}}}
    State: {'works': {'w': {'vars': {'counter': 3}}}}
    State: {'works': {'w': {'vars': {'counter': 3}}}}
    State: {'works': {'w': {'vars': {'counter': 4}}}}
    ...

----

*************************************************
Communication from LightningFlow to LightningWork
*************************************************

Communication from the LightningFlow (Flow) to the LightningWork (Work) while running **isn't supported yet**. If your application requires this feature, please open an issue on Github.

Here's an example of what would happen if you try to have the Flow communicate with the Work:

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

As you can see, there is a divergence between the values within the Work and the Flow.

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
