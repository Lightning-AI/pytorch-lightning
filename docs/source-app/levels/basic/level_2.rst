#########################
Level 2: Build a workflow
#########################
**Audience:** Users who need to build workflows.

**Prereqs:** You `read Lightning in 15 minutes <lightning_in_15_minutes.html>`_ and ran python code inside a LightningWork.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/workflow_intro.gif
    :alt: Animation showing how to convert your PyTorch code to LightningLite.
    :width: 800
    :align: center

----

*******************
What is a workflow?
*******************
A workflow coordinates 2 or more python scripts together. We call a workflow built with Lightning a *Lightning App*.

In this guide, we'll build a workflow in <5 minutes and explain how it works.

.. note:: If you've used other workflow tools for Python, in `level 4 <level_4.html>`_, we'll 
        generalize simple workflows to reactive workflows that allow you to build complex
        systems without much effort!

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
            self.work_A.run("running code A on a CPU machine")
            self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())

Run the app to see both works execute on separate machines 🤯

.. code:: python

    lightning run app app.py --cloud

And run it locally to see that it runs on your laptop without code changes 🤯🤯

.. code:: python

    lightning run app app.py

Now you can develop distributed cloud workflows on your laptop 🤯🤯🤯🤯!


----

**************************
Now you know ...
**************************

-------------
Orchestration
-------------

In these lines, you defined a LightningFlow which coordinates how the LightningWorks interact together.
In engineering, we call this **orchestration**:

.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-4">

        <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/orchestration.gif" width="100%">

.. raw:: html

        </div>
        <div class="col-md-8">

.. code:: python
    :emphasize-lines: 8, 15

    # app.py
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
            self.work_A.run("running code A on a CPU machine")
            self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())

.. raw:: html

        </div>
        </div>
    </div>

⚡⚡ Now you're an expert at orchestration!

.. hint::

    If you've used other orchestration frameworks before, this should already be familiar! In `level 4 <level_4.html>`_, you'll
    see how to generalize beyond "orchestrators" with reactive workflows that allow you to build complex
    systems without much effort!

----

---------------------------
Distributed cloud computing
---------------------------
The two pieces of independent Python code ran on *separate* 🤯🤯 machines:


.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-4">
        <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/distributed_computing.gif" width="100%">

.. raw:: html

        </div>
        <div class="col-md-8">

.. code:: python
    :emphasize-lines: 13, 16

    # app.py
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
            self.work_A.run("running code A on a CPU machine")
            self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())

.. raw:: html

        </div>
        </div>
    </div>

⚡⚡ Now you're a distributed computing wiz!

----

---------------------------
Multi-machine communication
---------------------------
Notice that the LightningFlow sent the variables: (**message_a** -> machine A),  (**message_b** -> machine B):

.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-4">
        <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/multi_machine_comms.gif" width="100%">

.. raw:: html

        </div>
        <div class="col-md-8">

.. code:: python
    :emphasize-lines: 15, 16, 17, 18

    # app.py
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
            message_a = "running code A on a CPU machine"
            message_b = "running code B on a GPU machine"
            self.work_A.run(message_a)
            self.work_B.run(message_b)

    app = L.LightningApp(LitWorkflow())


.. raw:: html

        </div>
        </div>
    </div>
⚡⚡ Now you're also an expert in networking and cross-machine communication!

----

-----------------------------
Multi-cloud and multi-cluster
-----------------------------
The full workflow (which we call a Lightning App), can easily be moved across clouds and clusters.

.. raw:: html

    <div class="display-card-container">
        <div class="row">
        <div class="col-md-4">
        <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/multi_cloud.gif" width="100%">

.. raw:: html

        </div>
        <div class="col-md-8">

.. code:: python
    :emphasize-lines: 12, 13, 14, 15

    # app.py
    import lightning as L

    class LitWorker(L.LightningWork):
        def run(self, message):
            print(message)

    class LitWorkflow(L.LightningFlow):
        def __init__(self, cloud) -> None:
            super().__init__()

            # run on cluster-A  OR  cluster-B
            cloud = 'cluster-A'
            comp_A = L.CloudCompute('cpu', clouds=[cloud])
            comp_B = L.CloudCompute('cpu', clouds=[cloud])
            self.work_A = LitWorker(cloud_compute=comp_A)
            self.work_B = LitWorker(cloud_compute=comp_B)

        def run(self):
            self.work_A.run("running code A on a CPU machine")
            self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())


.. raw:: html

        </div>
        </div>
    </div>
⚡⚡ Now your workflows are multi-cloud!

.. collapse:: Create a cluster on your AWS account

   |
   To run on your own AWS account, first `create an AWS ARN <../glossary/aws_arn.rst>`_.

   Next, set up a Lightning cluster (here we name it **cluster-A**):

   .. code:: bash

      # TODO: need to remove  --external-id dummy --region us-west-2
      lightning create cluster cluster-A --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc

   Run your code on the **cluster-A** cluster by passing it into CloudCompute:

   .. code:: python 

      compute = L.CloudCompute('gpu', clusters=['cluster-A'])
      app = L.LightningApp(LitWorker(cloud_compute=compute))

   .. warning:: 
      
      This feature is available only under early-access. Request access by emailing upgrade@lightning.ai.

----

.. collapse:: Kubernetes

    ABC 

.. collapse:: reproducibility

    ABC 

.. collapse:: Fault-tolerance

    ABC 

.. collapse:: Ran in a secure environment

    ABC 

----

***********************
Use Python control flow
***********************
Lightning code is simply **organized python**. If you know python, you already know Lightning. Use for-loops, if statements, while loops, timers, etc... as you do with Python:

.. code:: python
    :emphasize-lines: 2, 13, 16, 17, 21, 22

    import lightning as L
    from datetime import datetime

    class LitWorker(L.LightningWork):
        def run(self, message):
            print(message)

    class LitWorkflow(L.LightningFlow):
        def __init__(self) -> None:
            super().__init__()
            self.work_A = LitWorker(cloud_compute=L.CloudCompute('cpu'))
            self.work_B = LitWorker(cloud_compute=L.CloudCompute('gpu'))
            self._start_time = None

        def run(self):
            if self._start_time is None:
                self._start_time = datetime.now()
            self.work_A.run("running code A on a CPU machine")

            # start B, 5 seconds after A has finished
            elapsed_seconds = (datetime.now() - self._start_time).seconds
            if elapsed_seconds > 5:
                self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())

----

*************
Schedule work
*************
Although you can use python timers to scheduler work, 
Lightning has an optional shorthand API (`self.schedule <../../core_api/lightning_flow.html#lightning_app.core.flow.LightningFlow.schedule>`_) 
that uses `crontab syntax <https://crontab.guru/>`_:

.. code:: python
    :emphasize-lines: 17

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
            self.work_A.run("running code A on a CPU machine")

            # B runs once, and then again every hour
            if self.schedule("hourly"):
                self.work_B.run("running code B on a GPU machine")

    app = L.LightningApp(LitWorkflow())

----

*************************
Next step: A real example
*************************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 3: Communicate between works
   :description: Move variables and files across works.
   :col_css: col-md-3
   :button_link: level_3.html
   :height: 150
   :tag: beginner

.. displayitem::
   :header: Level 3: Communicate between works
   :description: Move variables and files across works.
   :col_css: col-md-3
   :button_link: level_3.html
   :height: 150
   :tag: beginner

.. displayitem::
   :header: Level 3: Communicate between works
   :description: Move variables and files across works.
   :col_css: col-md-3
   :button_link: level_3.html
   :height: 150
   :tag: beginner

.. displayitem::
   :header: Level 3: Communicate between works
   :description: Move variables and files across works.
   :col_css: col-md-3
   :button_link: level_3.html
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>