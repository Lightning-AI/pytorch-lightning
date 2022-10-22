#######################
Lightning in 15 minutes
#######################

**Required background:** Basic Python familiarity.

**Goal:** In this guide you'll learn the basic concept to develop with Lightning.

.. join_slack::
   :align: left

----

******************
What is Lightning?
******************
Lightning is an `open-source <https://github.com/Lightning-AI/lightning>`_ framework that provides **minimal organization to Python code** to develop workflows that
`run on your own AWS account <#run->`_, the `Lightning Cloud (fully-managed AWS) <https://lightning.ai/>`_ or `your own hardware <?>`_.

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

   class LitWorker(L.LightningWork):
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
   app = L.LightningApp(LitWorker(cloud_compute=compute))


**Lightning runs the same on the cloud and locally.**

Run on a GPU in your own AWS account or Lightning Cloud (fully-managed AWS):

.. code:: python

   lightning run app.py --cloud

Run on your own hardware:

.. code:: python 
   
   lightning run app.py

----

************
Key features
************
You now know enough to build pretty powerful cloud workflows. Here are some features available
to super-charge your work. 

# TODO: example of complex apps

----

----------------------------
Use different cloud machines
----------------------------
Change the cloud machine easily with CloudCompute:

.. code:: python

   
   compute = L.CloudCompute('default')          # 1 CPU
   compute = L.CloudCompute('cpu-small')        # 2 CPUs
   compute = L.CloudCompute('cpu-medium')       # 8 CPUs
   compute = L.CloudCompute('gpu')              # 1 T4 GPU
   compute = L.CloudCompute('gpu-fast')         # 1 V100 GPU
   compute = L.CloudCompute('gpu-fast-multi')   # 4 V100 GPU
   app = L.LightningApp(LitWorker(cloud_compute=compute))

More machine types are available when you `run on your AWS account <??>`_.

----

----------
Save money
----------
Lightning code is optimized to use cloud resources very efficiently. Here are a few optimizations you can enable:

Turn off the machine when it's idle with **idle_timeout**:

.. code:: python

   # IDLE TIME-OUT 

   # turn off machine when it's idle for 10 seconds
   compute = L.CloudCompute('gpu', idle_timeout=10)
   app = L.LightningApp(LitWorker(cloud_compute=compute))


Cloud machines are subject to availability in the cloud provider. Set a **wait_timeout** limit to how long you want to wait for a machine to start:

.. code:: python

   # WAIT TIME-OUT 
   
   # if the machine hasn't started after 60 seconds, cancel the work
   compute = L.CloudCompute('gpu', wait_timeout=60)
   app = L.LightningApp(LitWorker(cloud_compute=compute)

Use machines at a ~90% discount with **preemptible**: Pre-emptible machines are ~90% cheaper because they can be turned off at any second without notice:

.. code:: python
   
   # PRE-EMPTIBLE MACHINES

   # ask for a preemptible machine
   # wait 60 seconds before auto-switching to a full-priced machine
   compute = L.CloudCompute('gpu', preemptible=True, wait_timeout=60)
   app = L.LightningApp(LitWorker(cloud_compute=compute)

----

-----------------------
Run on your AWS account
-----------------------
To run on your own AWS account, first `create an AWS ARN <../glossary/aws_arn.rst>`_.   

Next, set up a Lightning cluster (here we name it pikachu):

.. code:: bash

   # TODO: need to remove  --external-id dummy --region us-west-2
   lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc

Run your code on the pikachu cluster by passing it into CloudCompute:

.. code:: python 

   compute = L.CloudCompute('gpu', clusters=['pikachu'])
   app = L.LightningApp(LitWorker(cloud_compute=compute))

.. warning:: 
   
   This feature is available only under early-access. Request access by emailing upgrade@lightning.ai.

----

----------------------
Use a custom container
----------------------
Run your cloud Lightning code with a custom container image by using **cloud_build_config**:

.. code:: python 
   
   # USE A CUSTOM CONTAINER

   cloud_config = L.BuildConfig(image="gcr.io/google-samples/hello-app:1.0")
   app = L.LightningApp(LitWorker(cloud_build_config=cloud_config))

----

--------------------------
Work with massive datasets
--------------------------
A LightningWork might need a large working folder for certain workloads such as ETL pipelines, data collection, training models and processing datasets.

Attach a disk up to 64 TB with **disk_size**:

.. code:: python

   # MODIFY DISK SIZE 

   # use 100 GB of space on that machine (max size: 64 TB)
   compute = L.CloudCompute('gpu', disk_size=100)
   app = L.LightningApp(LitWorker(cloud_compute=compute)

.. note:: when the work finishes executing, the disk will be deleted.

----

-------------------
Mount cloud folders
-------------------
To mount an s3 folder, use **Mount**:

.. code:: python

   # TODO: create a public demo folder
   # public bucket
   mount = Mount(source="s3://lightning-example-public/", mount_path="/foo")
   compute = L.CloudCompute(mounts=mount)

   app = L.LightningApp(LitWorker(cloud_compute=compute))

Read and list the files inside your LightningWork:

.. code:: python

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
      def run(self):
         os.listdir('/foo')
         file = os.file('/foo/a.jpg')

   app = L.LightningApp(LitWorker())

.. note::

   To attach private s3 buckets, sign up for our early access: support@lightning.ai.

----

***************************
Next step: Multiple Workers
***************************
In this guide, we showed how to run a single piece of code on a toy example. Check out these 
non-toy examples.

- A 
- B
- C

In the next guide, we'll learn how to run multiple LightningWork together


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Next step: Multiple workers
   :description: Run multiple LightningWorks together 
   :col_css: col-md-12
   :button_link: ../model/build_model_advanced.html#manual-optimization
   :height: 150
   :tag: beginner

.. raw:: html

        </div>
    </div>
