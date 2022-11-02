

***********
The toy app
***********
In this guide, we'll use the following toy app to illustrate the ideas:

.. code:: python

    # app.py
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self, message):
            for i in range(100000000000):
                print(message, i)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'))
            self.work_B = LitWorker(cloud_compute=L.CloudCompute('cpu'))

        def run(self):
            self.work_A.run("machine A counting")
            self.work_B.run("machine B counting")

    app = L.LightningApp(LitWorkflow())

----

***************************************
When to run a LightningWork in parallel
***************************************
Run LightningWork in parallel when you want to execute work in the background or at the same time as another work.
An example of when this comes up in machine learning is when data streams-in while a model trains.

----

********************
Run work in parallel
********************
By default, a LightningWork must complete before the next one runs:

.. code:: python
    :emphasize-lines: 18

    # app.py
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self, message):
            for i in range(100000000000):
                print(message, i)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'))
            self.work_B = LitWorker(cloud_compute=L.CloudCompute('cpu'))

        def run(self):
            self.work_A.run("machine A counting")

            # Work A must complete before B starts... which will take a while
            self.work_B.run("machine B counting")

    app = L.LightningApp(LitWorkflow())

|

Set (**parallel=True**) to run work_A in parallel so we can immediately start work B without waiting for A to complete:

.. code:: python
    :emphasize-lines: 12

    # app.py
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self, message):
            for i in range(100000000000):
                print(message, i)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'), parallel=True)
            self.work_B = LitWorker(cloud_compute=L.CloudCompute('cpu'))

        def run(self):
            self.work_A.run("machine A counting")
            self.work_B.run("machine B counting")

    app = L.LightningApp(LitWorkflow())
