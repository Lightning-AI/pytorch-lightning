###########################################
Level 6: Share variables between components
###########################################
**Audience:** Users who want to share variables and files across Lightning components.

**Prereqs:** You must have finished `intermediate level 5+ <run_lightning_work_in_parallel.rst>`_.

----

****************************************
Send a variable from Flow to a Component
****************************************

Todo

----

**************************************
Send a variable between two components
**************************************
Works cannot communicate directly between each other. Instead, a shared parent Flow must manage the communication.

# A needs to know something about B, maybe the time?

----

********************************************
Send a large variable between two components
********************************************
Payload.






















***********
The toy app
***********
In this page, we'll be using the following toy snippet:

.. code:: python

    # app.py
    import lightning as L

    class CountingWork(L.LightningWork):
        def __init__(self):
            super().__init__(parallel=True)
            self.count = 0

        def run(self):
            for i in range(int(1000000)):
                self.count += 1

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.counter = CountingWork(cloud_compute=L.CloudCompute('cpu'))

        def run(self):
            self.counter.run()
            count = self.counter.count
            print(count)

    app = L.LightningApp(LitWorkflow())

----

*****************************
Communicate from Flow to Work
*****************************
ABC

----

**********************************
Communicate between LightningWorks
**********************************
Works cannot communicate directly between each other. Instead, a shared parent Flow must manage the communication.

.. code:: python

    # app.py
    import lightning as L

    class CountingWork(L.LightningWork):
        def __init__(self):
            super().__init__(parallel=True)
            self.count = 0

        def run(self):
            for i in range(int(1000000)):
                self.count += 1

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.counter = CountingWork(cloud_compute=L.CloudCompute('cpu'))

        def run(self):
            self.counter.run()
            count = self.counter.count
            print(count)
