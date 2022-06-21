####################
Run work in parallel
####################

When there is a long-running workload such as a model training, or a deployment server, it may be desirable to run that work in parallel
while the rest of the app continues to execute.

----

****************************
Why do I need parallel work?
****************************
The default behavior of the ``LightningWork`` is to wait for the ``run`` method to complete:

.. code:: python

    import lightning_app as la


    class Root(lapp.LightningFlow):
        def __init__(self):
            self.work_component_a = lapp.demo.InfinteWorkComponent()

        def run(self):
            self.work_component_a.run()
            print("this will never print")

Since this Work component we created loops forever, the print statement will never execute. In practice
``LightningWork`` workloads are finite and don't run forever.

When a ``LightningWork`` performs a heavy operation (longer than 1 second), or requires its own hardware,
work that is *not* done in parallel will slow down your app.

----

********************
Enable parallel work
********************
To run work in parallel while the rest of the app executes without delays, enable ``parallel=True``:

.. code:: python
    :emphasize-lines: 5

    import lightning_app as la


    class Root(lapp.LightningFlow):
        def __init__(self):
            self.work_component_a = lapp.demo.InfinteWorkComponent(parallel=True)

        def run(self):
            self.work_component_a.run()
            print("repeats while the infinite work runs ONCE (and forever) in parallel")

Any work that will take more than **1 second** should be run in parallel
unless the rest of your app depends on the output of this work (for example, downloading a dataset).
