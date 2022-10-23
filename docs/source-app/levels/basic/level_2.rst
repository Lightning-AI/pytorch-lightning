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

And run it locally to see that it runs on your laptop without code changes 🤯🤯

.. code:: python

    lightning run app app.py

Now you can develop distributed cloud workflows on your laptop 🤯🤯🤯🤯!

----

***********************
Now you're an expert in
***********************

.. collapse:: Orchestration

    |

    In these lines, you defined a LightningFlow which coordinates how the LightningWorks interact together.
    In engineering, we call this **orchestration**:

    .. code:: python
        :emphasize-lines: 9, 16

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

            # the run method of LightningFlow is an orchestrator
            def run(self):
                self.work_A.run("python code A running on a CPU machine")
                self.work_B.run("python code B running on a GPU machine")

        app = L.LightningApp(LitWorkflow())


.. collapse:: Distributed computing

    |

    The two pieces of independent Python code ran on *separate* 🤯🤯 machines:

    .. code:: python
        :emphasize-lines: 14, 17

        # app.py
        # MULTIPLE WORKERS
        import lightning as L

        class LitWorker(L.LightningWork):
            def run(self, message):
                print(message)

        class LitWorkflow(L.LightningFlow):
            def __init__(self) -> None:
                super().__init__()

                # runs on machine 1
                self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'))

                # runs on machine 2
                self.work_B = LitWorker(cloud_compute=L.CloudCompute('gpu'))

            def run(self):
                self.work_A.run("python code A running on a CPU machine")
                self.work_B.run("python code B running on a GPU machine")

        app = L.LightningApp(LitWorkflow())
    
    Now you're a distributed computing wiz!


.. collapse:: Multi-machine communication

    |

    Notice that the LightningFlow sent the variables: (**message_a** -> machine A),  (**message_b** -> machine B):

    .. code:: python
        :emphasize-lines: 16, 17, 18, 19

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
                message_a = "python code A running on a CPU machine"
                message_b = "python code B running on a CPU machine"
                self.work_A.run(message_a)
                self.work_B.run(message_b)

        app = L.LightningApp(LitWorkflow())
    
    Now you're also an expert in networking and cross-machine communication!


.. collapse:: Multi-cloud

    ABC 

.. collapse:: Kubernetes

    ABC 

.. collapse:: reproducibility

    ABC 

.. collapse:: Fault-tolerance

    ABC 

.. collapse:: Ran in a secure environment

    ABC 

----

*****************************
Next step: Reactive Workflows
*****************************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 3: Build a reactive workflow
   :description: Coordinate complex logic simply.
   :col_css: col-md-12
   :button_link: level_3.html
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>