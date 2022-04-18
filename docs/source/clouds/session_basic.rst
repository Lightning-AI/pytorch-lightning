:orphan:

.. _grid_cloud_session_basic:

##########################
Train on the cloud (basic)
##########################
**Audience**: Anyone looking to train across many machines at once on the cloud.

----

*****************************
Why do I need cloud training?
*****************************
Training on the cloud is a cost effective way to train your models faster by allowing you to access powerful GPU machines.

For example, if your model takes 10 days to train on a CPU machine, here's how cloud training can speed up your training time:

.. list-table:: Training speed vs cost
   :widths: 20 20 20
   :header-rows: 1

   * - Machine type
     - Training time
     - Cost (AWS 1 M60 GPU)
   * - CPU
     - 10 days
     - $12.00
   * - 1 GPU
     - 2 days
     - $11.52
   * - 2 GPU
     - 1 day
     - $20.64
   * - 4 GPU
     - 12 hours
     - $19.08

----

***********************************
Start a cloud machine in < 1 minute
***********************************
Lightning has a native cloud solution with various products (lightning-grid) designed for researchers and ML practicioners in industry.
To start an interactive machine simply go to `Lightning Grid <https://platform.grid.ai>`_ to create a free account, then start a new Grid Session.

A Grid Session is an interactive machine with 1-16 GPUs per machine.

.. image:: https://docs.grid.ai/assets/images/new-session-3c58be3fd64ffabcdeb7b52516e0782e.gif
    :alt: Start a Grid Session in a few seconds

----

*************************
Open the Jupyter Notebook
*************************
Once the Session starts, open a Jupyter notebook.

.. raw:: html

    <video width="100%" max-width="800px" controls muted playsinline
    src="https://grid-docs.s3.us-east-2.amazonaws.com/jupyter.mp4"></video>

----

************************
Clone and run your model
************************
On the Jupyter page you can use a Notebook, or to clone your code and run via the CLI.

.. raw:: html

    <video width="100%" max-width="800px" controls muted playsinline
    src="https://grid-docs.s3.us-east-2.amazonaws.com/notebook_or_cli.mp4"></video>

----

.. include:: grid_costs.rst

----

**********
Next Steps
**********
Here are the recommended next steps depending on your workflow.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Run a model in the background
   :description: Learn to run a model in the background
   :col_css: col-md-6
   :button_link: run_basic.html
   :height: 180
   :tag: basic

.. displayitem::
   :header: Run with your own cloud credentials
   :description: Learn how to use Grid products on your Company or University private cloud account.
   :col_css: col-md-6
   :button_link: run_expert.html
   :height: 180
   :tag: expert

.. raw:: html

        </div>
    </div
