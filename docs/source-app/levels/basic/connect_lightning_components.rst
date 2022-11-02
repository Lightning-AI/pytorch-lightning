####################################################
Level 2: Connect components into a full stack AI app 
####################################################

**Audience:** Users who want to build apps with multiple components.

**Prereqs:** You know how to `build a component <build_a_lightning_component.html>`_.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/workflow_intro.gif
    :alt: Animation showing how to convert your PyTorch code to LightningLite.
    :width: 800
    :align: center

----

****************************
What is a full stack AI app?
****************************
A full stack AI app coordinates 2 or more `Lightning components <build_a_lightning_component.html>`_ together.
We call this system of components interacting with each other a Lightning App.

In this guide, we'll coordinate 2 components together and explain how it works.

.. note:: If you've used workflow tools for Python, this page describes conventional DAGs.
        In `level 5 <../intermediate/run_lightning_work_in_parallel.html>`_, we introduce reactive workflows that generalize beyond DAGs
        so you can build complex systems without much effort. 

----

***********
The toy app
***********

In this app, we define two components that run across 2 separate machines. One to train a model on a GPU machine and one to analyze the model 
on a separate CPU machine. We save money by stopping the GPU machine when the work is done.

.. lit_tabs::
   :titles: Import Lightning; Define Component 1;  Define Component 2; Orchestrator; Connect component 1; Connect component 2; Implement run; Train; Analyze; Define app placeholder
   :descriptions: First, import Lightning; This component trains a model on a GPU machine; This component analyzes a model on a CPU machine; Define the LightningFlow that orchestrates components; Component 1 will run on a CPU machine; Component 2 will run on an accelerated GPU machine; Describe the workflow in the run method; Training runs first and completes; Analyze runs after training completes; This allows the app to be runnable
   :code_files: ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py; ./level_2_scripts/hello_app.py
   :highlights: 2; 4-6; 8-10; 12; 15; 16; 18; 19; 20; 22
   :app_id: abc123
   :tab_rows: 4
   :height: 450px

|

Now run the app:

.. lit_tabs::
   :titles: Run on Lightning cloud; Your own hardware
   :descriptions: Run to see these 2 components execute on separate machines 🤯; Run it locally without code changes 🤯🤯;
   :code_files: ./hello_components/code_run_cloud.bash; ./hello_components/code_run_local.bash
   :tab_rows: 7
   :height: 195px

|

Now you can develop distributed cloud apps on your laptop 🤯🤯🤯🤯!


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

⚡⚡ Now you know how to orchestrate!

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

Run on Cluster A

.. code:: bash

    lightning run app app.py --cloud cluster-A

Run on Cluster B

.. code:: bash

    lightning run app app.py --cloud cluster-B

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
      
      This feature is available only under early-access. Request access by emailing support@lightning.ai.

----

----------
Kubernetes
----------
Under the hood, Lightning works with Kubernetes to manage the infrastructure on your behalf. 
This means you don't have to learn kubernetes to run cloud workflows anymore.

Lightning also plays well with current Kubernetes clusters and even lets you 
`create the clusters yourself with terraform <https://github.com/Lightning-AI/terraform-aws-lightning-byoc>`_.

----

-------------------
Secure environments
-------------------
When you build clusters with Lightning, we ensure everything is configured securily which includes abiding by SOC-2 (type 1) guidelines.

For startups or enterprises who want to learn more, please contact support@lightning.ai.

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
You now know enough to build pretty powerful cloud workflows!

Choose an example to walk through step-by-step. 

Once you feel comfortable with these examples, move to the intermediate guide, where we'll learn about reactive
workflows which will allow you build full-stack AI applications.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Example: Train PyTorch on the cloud
   :description: Train a PyTorch model in single or multi-node on the cloud
   :button_link: train_pytorch_on_the_cloud.html
   :col_css: col-md-6
   :height: 150
   :tag: basic

.. displayitem::
   :header: Example: Deploy a model API
   :description: Deploy a model behind a load-balanced API.
   :button_link: deploy_ai_model_api.html
   :col_css: col-md-6
   :height: 150
   :tag: basic

.. displayitem::
   :header: Example: Develop a Jupyter Notebook component
   :description: Develop a LightningWork that runs a notebook on the cloud.
   :button_link: run_jupyter_notebook_on_the_cloud.html
   :col_css: col-md-6
   :height: 150
   :tag: basic

.. displayitem::
   :header: Example: Create a model demo
   :description: Demo POCs and MVPs which can be shared with a public web user interface.
   :button_link: create_a_model_demo.html
   :col_css: col-md-6
   :height: 150
   :tag: basic

.. displayitem::
   :header: Example: Directed Acyclical Graph (DAG)
   :description: Learn how to build a DAG with Lightning workflows.
   :button_link: create_a_model_demo.html
   :col_css: col-md-6
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div>