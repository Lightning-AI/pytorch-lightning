############################################
Level 3: Run on your AWS account efficiently
############################################
**Audience:** Users who want to use the AWS cloud efficiently.

**Prereqs:** You must have finished the `Basic levels <../basic/>`_.

----

***********
The toy app
***********
In this page, we'll be using the following toy snippet:

.. code:: python

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
        def run(self):
            print("run ANY python code here")

    compute = L.CloudCompute('cpu')
    app = L.LightningApp(LitWorker(cloud_compute=compute))

----

***********************
Run on your AWS account
***********************

.. include:: ../basic/run_on_aws_account.rst

----

***********************
Auto-stop idle machines
***********************

**idle_timeout**: Turn off the machine when it's idle for n seconds.

.. code:: python

    # IDLE TIME-OUT 

    # turn off machine when it's idle for 10 seconds
    compute = L.CloudCompute('gpu', idle_timeout=10)
    app = L.LightningApp(LitWorker(cloud_compute=compute))

----

***************************
Auto-timeout submitted work
***************************

**wait_timeout**: Wait n seconds for machine to be allocated by the cloud provider before cancelling the work:

.. code:: python

    # WAIT TIME-OUT 
    
    # if the machine hasn't started after 60 seconds, cancel the work
    compute = L.CloudCompute('gpu', wait_timeout=60)
    app = L.LightningApp(LitWorker(cloud_compute=compute)

----

*********************************
Use spot machines (~70% discount)
*********************************
**spot**: Spot machines are ~70% cheaper because they can be turned off at any second without notice:

.. code:: python
    
    # ask for a spot machine
    # wait 60 seconds before auto-switching to a full-priced machine
    compute = L.CloudCompute('gpu', spot=True, wait_timeout=60)
    app = L.LightningApp(LitWorker(cloud_compute=compute)
