###########################################
Level 6: Share variables between components
###########################################
**Audience:** Users who want to share variables and files across Lightning components.

**Prereqs:** You must have finished `intermediate level 5+ <run_lightning_work_in_parallel.rst>`_.

----

****************************************
Send a variable from Flow to a Component
****************************************
When a variable is defined on the LightningFlow (orchestrator), and
then it's passed into functions for the work components, under the hood
Lightning sends the variables across the machines for you automatically.

.. lit_tabs::
   :descriptions: Toy app with a single work; Remember this component may live on its own machine; The flow may be on a separate machine as well; This variable is on the flow machine; When passed to the work component, it is actually sent across the network under the hood.; When it prints here, it prints on the work component machine (not the flow machine); The second string was directly created on machine 1
   :code_files: ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py
   :highlights: ; 4-7; 9-16; 15; 16; 6; 7;
   :app_id: abc123
   :tab_rows: 3
   :height: 430px

|

In this example, we learned that we can send variables to components like in regular Python.
On a local machine, it will behave like Python. When the workflow is distributed on the cloud,
it makes network calls under the hood, but still functions like Python to you.

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
