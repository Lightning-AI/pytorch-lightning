:orphan:

.. _quick_start:

############
Quick Start
############

In this guide, we'll run an application that trains
an image classification model with the `MNIST Dataset <https://en.wikipedia.org/wiki/MNIST_database>`_,
and uses `Gradio <https://gradio.app>`_ to serve it.

----

**********************
Step 1 - Installation
**********************

First, you'll need to install Lightning. You can find the complete guide here.

Then, you'll need to install the `Lightning Quick Start package <https://github.com/Lightning-AI/lightning-quick-start>`_.

.. code-block:: bash

    lightning install app lightning/quick-start

And download the training script used by the App:


----

**********************
Step 2 - Run the app
**********************

To run your app, copy the following command to your local terminal:

.. code-block:: bash

    lightning run app app.py

And that's it!

.. admonition::  You should see the app logs in your terminal.
   :class: dropdown

    .. code-block:: console

        Your Lightning App is starting. This won't take long.
        INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

        Global seed set to 42

        GPU available: True (mps), used: False
        TPU available: False, using: 0 TPU cores

          | Name    | Type     | Params | In sizes       | Out sizes
        ------------------------------------------------------------------
        0 | model   | Net      | 1.2 M  | [1, 1, 28, 28] | [1, 10]
        1 | val_acc | Accuracy | 0      | ?              | ?
        ------------------------------------------------------------------
        1.2 M     Trainable params
        0         Non-trainable params
        1.2 M     Total params
        Epoch 4: 100%|█████████████████████████| 16/16 [00:00<00:00, 32.31it/s, loss=0.0826, v_num=0]
        `Trainer.fit` stopped: `max_epochs=5` reached.

        Running on local URL:  http://127.0.0.1:62782/
        ...


The app will open your browser and show an interactive demo:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/qiuck-start-tensorboard-tab.png
    :alt: Quick Start UI - Model Training Tab
    :width: 100 %

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/quick-start-gradio-tab.png
    :alt: Quick Start UI - Interactive Demo Tab
    :width: 100 %

----

This app behind the scenes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This application has one flow component which coordinates two works executing their own python script.
Once the training is finished, the trained model weights are passed to the serve component.


Here is how the components of a Lightning app are structured:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/quick_start_components.gif
    :alt: Quick Start Application
    :width: 100 %

Here is the application timeline:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/timeline.gif
    :alt: Quick Start Timeline Application
    :width: 100 %

----

**************************************
Steps 3 - Build your app in the cloud
**************************************

Simply add ``--cloud`` to run this application in the cloud 🤯

.. code-block:: bash

    lightning run app app.py --cloud

Congratulations! You've now run your first application with Lightning.

----

***********
Next Steps
***********

To learn how to build and modify apps, go to the :ref:`basics`.

To learn how to create UIs for your apps, read :ref:`ui_and_frontends`.
