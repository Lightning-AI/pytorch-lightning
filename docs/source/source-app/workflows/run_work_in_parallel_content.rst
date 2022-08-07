

****************************************************
What running LightningWorks in parallel does for you
****************************************************

When there is a long-running workload such as a model training, or a deployment server, you might want to run that LightningWork in parallel
while the rest of the Lightning App continues to execute.

The default behavior of the ``LightningWork`` is to wait for the ``run`` method to complete:

.. code:: python

    import lightning as L


    class Root(L.LightningFlow):
        def __init__(self):
            self.work_component_a = L.demo.InfinteWorkComponent()

        def run(self):
            self.work_component_a.run()
            print("this will never print")

Since this LightningWork component we created loops forever, the print statement will never execute. In practice
``LightningWork`` workloads are finite and don't run forever.

When a ``LightningWork`` performs a heavy operation (longer than 1 second), or requires its own hardware,
LightningWork that is *not* done in parallel will slow down your app.

----

******************************
Enable parallel LightningWorks
******************************
To run LightningWorks in parallel, while the rest of the app executes without delays, enable ``parallel=True``:

.. code:: python
    :emphasize-lines: 5

    import lightning as L


    class Root(L.LightningFlow):
        def __init__(self):
            self.work_component_a = L.demo.InfinteWorkComponent(parallel=True)

        def run(self):
            self.work_component_a.run()
            print("repeats while the infinite work runs ONCE (and forever) in parallel")

Any LightningWorks that will take more than **1 second** should be run in parallel
unless the rest of your Lightning App depends on the output of this work (for example, downloading a dataset).
