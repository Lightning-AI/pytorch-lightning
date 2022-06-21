:orphan:

.. _grid_cloud_run_basic:

#################################
Train on the cloud (intermediate)
#################################
**Audience**: Anyone looking to train a model on the cloud in the background

----

****************************
What is background training?
****************************
Background training lets you train models in the background without you needing to interact with the machine. As the model trains you can monitor its progress via Tensorboard or an experiment manager of your choice.

----

*************************
0: Install lightning-grid
*************************
First Navigate to https://platform.grid.ai to create a free account.

Next, install lightning-grid and login

.. code:: bash

      pip install lightning-grid
      grid login

      # Login successful. Welcome to Grid.

----

*******************
1: Create a dataset
*******************
Create a datastore which optimizes your datasets for training at scale on the cloud. Datastores can be created from all sorts of sources such as .zip and .tar links, local files/folders and even s3 buckets.

Let's create a datastore from this .zip file

.. code:: bash

   grid datastore create https://pl-flash-data.s3.amazonaws.com/tinycifar5.zip --name cifar5

Now your dataset is ready to be used for training on the cloud!

.. note::  In some *research* workflows, your model script ALSO downloads the dataset. If the dataset is only a few GBs this is fine. Otherwise we recommend you create a Datastore.

----

**************************
2: Choose the model to run
**************************
You can run any python script in the background. For this example, we'll use a simple classifier:

Clone the code to your machine:

.. code:: bash

      git clone https://github.com/williamFalcon/cifar5-simple.git
      cd cifar5-simple

.. note:: Code repositories can be as complicated as needed. This is just a simple demo.

----

*******************
3: Run on the cloud
*******************
To run this model on the cloud with the attached datastore, use the **grid run** command:

.. code:: bash

      grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5

The grid command has two parts the *[run args]* and the *[file args]*

.. code:: bash

      grid run [run args] file.py [file args]

----

*********************
4: Monitor and manage
*********************
Now that your model is running in the background, `monitor and manage it here <https://platform.grid.ai/#/runs>`_.

You can also monitor its progress on the commandline:

.. code:: bash

      grid status

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
   :header: Run many models at once
   :description: Learn how to find the best performaning model by running multiple models at once using a sweep.
   :col_css: col-md-4
   :button_link: run_intermediate.html
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div
