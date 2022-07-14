:orphan:

.. _quick_start:

***********
Quick Start
***********

In this guide, we'll run an application that trains
an image classification model with the `MNIST Dataset <https://en.wikipedia.org/wiki/MNIST_database>`_,
and uses `FastAPI <https://fastapi.tiangolo.com/>`_ to serve it.

Step 1 - Installation
=====================

First, you'll need to install Lightning from source. You can find the complete guide here: :ref:`install`.

Then, you'll need to install the `Lightning Quick Start package <https://github.com/PyTorchLightning/lightning-quick-start>`_.

.. code-block:: bash

    lightning install quick-start

And download the training script used by the App:

.. code-block:: bash

    curl https://gist.githubusercontent.com/tchaton/b81c8d8ba0f4dd39a47bfa607d81d6d5/raw/8d9d70573a006d95bdcda8492e798d0771d7e61b/train_script.py > train_script.py

Step 2 - Run the app
====================

To run your app, copy the following command to your local terminal:

.. code-block:: bash

    lightning run app app.py

And that's it!

.. admonition::  You should see the app logs in your terminal.
   :class: dropdown

    .. code-block:: console

        INFO: Your app has started. View it in your browser: http://http://127.0.0.1:7501

        INFO: Running train_script: .../lightning/demo/quick_start/train/train.py
        Global seed set to 42
        GPU available: False, used: False
        TPU available: False, using: 0 TPU cores
        IPU available: False, using: 0 IPUs
        HPU available: False, using: 0 HPUs

        | Name    | Type     | Params
        -------------------------------------
        0 | model   | Net      | 1.2 M
        1 | val_acc | Accuracy | 0
        -------------------------------------
        1.2 M     Trainable params
        0         Non-trainable params
        1.2 M     Total params
        4.800     Total estimated model params size (MB)
        Epoch 3: 100%|█████████████████████████| 8/8 [00:03<00:00,  2.35it/s, loss=1.58, v_num=39]

        INFO: Running serve_script: .../lightning/demo/quick_start/serve/serve.py
        INFO: INFO:     Started server process [4808]
        INFO: INFO:     Waiting for application startup.
        INFO: INFO:     Application startup complete.
        INFO: INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
        ...


The build command will launch the app admin panel UI. In your app admin, you can track your app's progress, or click on the **Open App** button to view see your app's UI:

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/quick_start_ui.png
    :alt: Quick Start UI
    :width: 100 %

This app behind the scenes
--------------------------

This application has one flow component which coordinates two works executing their own python script.
Once the training is finished, the trained model weights are passed to the serve component.


Here is how the components of a Lightning app are structured:

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/quick_start_components.gif
    :alt: Quick Start Application
    :width: 100 %

Here is the application timeline:

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/timeline.gif
    :alt: Quick Start Timeline Application
    :width: 100 %


Steps 3 - Build your app in the cloud
=====================================

Simply add **--cloud** to run this application in the cloud 🤯

.. code-block:: bash

    lightning run app app.py --cloud

And with just one line of code, run on cloud GPUs!

.. code-block:: bash

    USE_GPU=1
    lightning run app app.py --cloud

Congratulations! You've now run your first application with Lightning.


Next steps
==========

To learn how to build and modify apps, go to the :ref:`basics`.

To learn how to create UIs for your apps, read :ref:`ui_and_frontends`.
