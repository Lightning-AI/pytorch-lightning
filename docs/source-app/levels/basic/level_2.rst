#########################
Level 2: Build a workflow
#########################
**Audience:** Users who need to build workflows.

**Prereqs:** You `read Lightning in 15 minutes <lightning_in_15_minutes.html>`_ and ran python code inside a LightningWork.

----

*******************
What is a workflow?
*******************
A workflow coordinates 2 or more python scripts together. We call a workflow built with Lightning a *Lightning App*.

In this guide, we'll build a workflow in <5 minutes and explain how it works.

----

**************
A Toy Workflow
**************

[VIDEO showing this]

[BUTTON TO DEPLOY THIS EXAMPLE]

In the previous example, we defined this LightningWork that can run ⚡ *any* ⚡ piece of Python code:

.. code:: python 

    # app.py
    # SINGLE WORKER
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self):
            print("ANY python code can run here")

    # uses 1 cloud GPU (or your own hardware)
    compute = L.CloudCompute('gpu')
    app = L.LightningApp(LitWorker(cloud_compute=compute))

In this example, let's run two pieces of Python code in a workflow:

.. code:: python

    # app.py
    # MULTIPLE WORKERS
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self, message):
            print(message)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'))
            self.work_B = LitWorker(cloud_compute=L.CloudCompute('gpu'))

        def run(self):
            self.work_A.run("python code A running on a CPU machine")
            self.work_B.run("python code B running on a GPU machine")

    app = L.LightningApp(LitWorkflow())

Run the app to see both works execute on separate machines 🤯

.. code:: python

    lightning run app app.py --cloud

And run it locally to see that it runs on your laptop 🤯🤯

.. code:: python

    lightning run app app.py

Now you can develop distributed cloud workflows on your laptop 🤯🤯🤯🤯!

----
