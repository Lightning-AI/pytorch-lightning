:orphan:

.. _grid_cloud_advanced:

#############################
Train on the cloud (advanced)
#############################
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

----

*******************
1: Create a dataset
*******************
Create a datastore which optimizes your datasets for training at scale on the cloud.

First, let's download a dummy dataset we created.

.. code:: bash

      # download
      curl https://pl-flash-data.s3.amazonaws.com/cifar5.zip -o cifar5.zip

      # unzip
      unzip cifar5.zip

Now create the datastore

.. code:: bash

      grid datastore create cifar5/ --name cifar5

Now your dataset is ready to be used for training on the cloud!

.. note::  In some *research* workflows, your model script ALSO downloads the dataset. If the dataset is only a few GBs this is fine. Otherwise we recommend you create a Datastore.

----

**************************
2: Choose the model to run
**************************
You can run any python script in the background. For this example, we'll use a simple classifier:

Clone the code to your machine:

.. code bash

      git clone https://github.com/williamFalcon/cifar5-simple.git


.. note:: Code repositories can be as complicated as needed. This is just a simple demo.

----

*******************
3: Run on the cloud
*******************
To run this model on the cloud, use the **grid run** command which has two parts:

.. code:: bash

      grid run [run args] file.py [file args]

To attach the datastore **cifar5** to the **cifar5.py** file use the following command:

.. code:: bash

      # command | the datastore to use   |  the model  | argument to the model
      grid run --datastore_name cifar5 cifar5.py.py --data_dir /datastores/cifar5

----

*********************
4: Monitor and manage
*********************
Now that your model is running in the background you can monitor and manage it `here <https://platform.grid.ai/#/runs>`_.

You can also monitor its progress on the commandline:

.. code:: bash

      grid status

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
   :description: Learn how to run many models at once using sweeps.
   :col_css: col-md-12
   :button_link: session_intermediate.html
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div
