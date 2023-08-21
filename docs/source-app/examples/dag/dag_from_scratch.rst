:orphan:

###################################################
Scheduled DAG with pandas and sklearn from scratch.
###################################################

**Audience:** Users coming from MLOps to Lightning Apps, looking for more flexibility.

**Level:** Intermediate.

In this example, you will learn how to create a simple DAG which:

* Download and process some data
* Train several models and report their associated metrics

and learn how to schedule this entire process.

Find the complete example `here <https://github.com/Lightning-AI/lightning/blob/master/examples/app/dag/app.py>`_.

----

**************************
Step 1: Implement your DAG
**************************

Here is an illustration of the DAG to implement:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/simple_dag.png
    :alt: Simple DAG
    :width: 100 %

First, let's define the component we need:

* DataCollector is responsible to download the data
* Processing is responsible to execute a ``processing.py`` script.
* A collection of model work to train all models in parallel.

.. literalinclude:: ../../../../examples/app/dag/app.py
    :lines: 52-74

And its run method executes the steps described above.

.. literalinclude:: ../../../../examples/app/dag/app.py
    :lines: 76-99

----

*****************************
Step 2: Define the scheduling
*****************************

.. literalinclude:: ../../../../examples/app/dag/app.py
    :lines: 102-130
