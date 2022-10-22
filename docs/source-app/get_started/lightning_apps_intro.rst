############################
Lightning Apps in 15 minutes
############################

**Required background:** Basic Python familiarity.

**Goal:** In this guide you'll learn the basic concepts to develop a Lightning App.

.. join_slack::
   :align: left

----

******************
What is Lightning?
******************
Lightning is an `open-source <https://github.com/Lightning-AI/lightning>`_ framework that provides **minimal organization to Python code** to compose workflows that
run on *your own AWS account*, the `Lightning Cloud (fully-managed AWS) <https://lightning.ai/>`_ or your own hardware.

.. note:: You don't need to know PyTorch or PyTorch Lightning to use Lightning.

----

*************************
Step 1: Install Lightning
*************************
.. code:: bash

    python -m pip install -U lightning

----

***************************
Step 2: Run any python code
***************************
[video showing this]

Lightning organizes Python code. Drop any piece of code into the LightningWork class and run on the cloud or your own hardware:

.. code:: python

   # app.py
   import lightning as L

   class AnyPythonCode(L.LightningWork):
      def run(self):
         message = """
         ANY python code can run here such as:
         - train a model
         - launch a deployment server
         - label data
         - run a react app, dash app, streamlit app, etc...
         - start a jupyter notebook
         - subprocess.Popen('echo run any shell script, python scripts or non python files')
         """
         print(message)

   # uses 1 cloud GPU (or your own hardware)
   compute = L.CloudCompute('gpu')
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute))


**Lightning runs the same on the cloud and locally.**

Run on a GPU in your own AWS account or Lightning Cloud (fully-managed AWS):

.. code:: python

   lightning run app.py --cloud

Run on your own hardware:

.. code:: python 
   
   lightning run app.py

----

**********
Save money
**********
Lightning code is optimized to use cloud resources very efficiently. Here are a few optimizations you can enable:

Turn off the machine when it's idle with **idle_timeout**:

.. code:: python

   # IDLE TIME-OUT 

   # turn off machine when it's idle for 10 seconds
   compute = L.CloudCompute('gpu', idle_timeout=10)
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute))


Cloud machines are subject to availability in the cloud provider. Set a **wait_timeout** limit to how long you want to wait for a machine to start:

.. code:: python

   # WAIT TIME-OUT 
   
   # if the machine hasn't started after 60 seconds, cancel the work
   compute = L.CloudCompute('gpu', wait_timeout=60)
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute)

Use machines at a ~90% discount with **pre-emptible**: Pre-emptible machines are ~90% cheaper because they can be turned off at any second without notice:

.. code:: python
   
   # PRE-EMPTIBLE INSTANCES

   # ask for a preemptible machine
   # wait 60 seconds before auto-switching to a full-priced machine
   compute = L.CloudCompute('gpu', preemptible=True, wait_timeout=60)
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute)

Don't pay for disk space you don't need. Configure it with **disk_size**

.. code:: python

   # MODIFY DISK SIZE 

   # use 10 GB of space on that machine
   compute = L.CloudCompute('gpu', disk_size=10)
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute)

----

***********************
Run on your AWS account
***********************
To run on your own AWS account, set up a Lightning cluster (here we name it pikachu):

.. code:: bash

   lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc --external-id dummy --region us-west-2

Run your code on the pikachu cluster by passing it into CloudCompute:

.. code:: python 

   compute = L.CloudCompute('gpu', clusters=['pikachu'])
   app = L.LightningApp(AnyPythonCode(cloud_compute=compute))

.. hint:: 

   Follow `this guide <??>`_ to create your AWS arn and external-id.



----


*******************
Mount cloud folders
*******************

disk_size

----

****************
Own docker image
****************
D7F5D5

----



**Why should I use Lightning?**

The Lightning standard has proven to be a succesful because the Lightning structure
allows teams and solo developers to organize their Python code which:

- 10x development speed 
- structural modularity
- standard interface
- built-in fault-tolerance and observability
- full flexibility

These elements allow teams and solo developers to move lightning fast through project implementations.

[TODO: graphic]

**I don't have time to learn a new library**

We built Lightning because we hate learning new frameworks. It's designed to be a very thin organizational layer for Python. 

A 10 minute investment to learn the 2 core principles of Lightning will save your hundreds of hours of not having to learn:

- kubernetes
- dag-systems
- YAML 
- distributed programming
- fault tolerance
- state management
- cross-machine communication
- distributed file-system management
- ... and much more 

etc... 

It's kind of like learning to drive a car so you don't have to learn how physics works, 
fuild dynamics, combustion engines and how to make your own gasoline.

*************
More examples
*************
Build more advanced apps with the following examples.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Build an ML product
   :description: Build an app to caption sound.
   :col_css: col-md-4
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :height: 150
   :tag: beginner

.. displayitem::
   :header: Train a model continuously
   :description: Train a model repeatedly with streaming data.
   :col_css: col-md-4
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :height: 150
   :tag: beginner

.. displayitem::
   :header: Deploy a load-balanced model
   :description: Deploy a model with a custom load-balancing rule
   :col_css: col-md-4
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :height: 150
   :tag: intermediate

.. raw:: html

        </div>
    </div>
