#################################################
Level 7: Communicate between Lightning Components
#################################################
**Audience:** Users who want to share variables and files across LightningWorks

**Prereqs:** You must have finished `intermediate level 5+ <run_lightning_work_in_parallel.rst>`_.

----

*****************************
Communicate from Work to Flow
*****************************
In this guide, we'll use the following toy app to illustrate the ideas:

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


**************************
Why you need communication
**************************
When two works are running 


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

    class PrintingWork(L.LightningWork):
        def run(self, message):
            print(message)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.counter = CountingWork(cloud_compute=L.CloudCompute('cpu'))
            self.printer = PrintingWork(cloud_compute=L.CloudCompute('gpu'))

        def run(self):
            self.counter.run()
            self.work_B.run("running code B on a GPU machine")