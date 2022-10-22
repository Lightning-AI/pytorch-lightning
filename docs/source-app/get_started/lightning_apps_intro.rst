############################
Lightning Apps in 15 minutes
############################

**Required background:** Basic Python familiarity.

**Goal:** In this guide you'll learn the basic concepts to develop a Lightning App.

.. join_slack::
   :align: left

******************
What is Lightning?
******************
Lightning is an open-source framework to develop workflows which can run locally or on the cloud. Lightning
is a thin-organizational layer on top of Python which means you don't have to learn a new framework.


**PyTorch Lightning vs Lightning**  

Lightning was born out of PyTorch Lightning. You do not need to know PyTorch Lightning or anything about
machine learning to use Lightning.

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

----

************
The toy code
************
[video showing this]

In the next 2 minutes we will run the following toy code to understand how Lightning works.

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
   app = L.LightningApp(AnyPythonCode(cloud_compute=L.CloudCompute('gpu')))


**Lightning runs the same on the cloud and locally.**

Run on a GPU in your own AWS account or Lightning Cloud (fully-managed AWS):

.. code:: python

   lightning run app.py --cloud

Run on your own hardware:

.. code:: python 
   
   lightning run app.py

----

*************************
Step 0: Install Lightning
*************************

.. code:: bash

    python -m pip install -U lightning

***********************************
Step 1: Find a piece of python code
***********************************
Pick any arbitrary piece of python code:

.. code:: python

   





************************
What is a Lightning App?
************************
A Lightning app is a simple way to define a distributed, complex cloud app for machine learning.
These applications require coordination of complex cloud resources such as high-speed disks for data loading,
model training, deployment servers, load-balancers, and more. The interaction between these components
happens in simple python instead of dozens of complex YAML files.

----

***********
The XYZ app
***********
The first app we'll build is a fun ML product to annotate speech.

----

*************************
Step 1: Install Lightning
*************************
Activate your `virtual environment <install_beginner.rst>`_ and run this command:

.. code:: bash

    python -m pip install -U lightning


----

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
