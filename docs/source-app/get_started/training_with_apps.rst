:orphan:

################################
Evolve a model into an ML system
################################

.. _convert_pl_to_app:

**Required background:** Basic Python familiarity and complete the :ref:`build_model` guide.

**Goal:** We'll walk you through the two key steps to build your first Lightning App from your existing PyTorch Lightning scripts.


*******************
Training and beyond
*******************

With `PyTorch Lightning <https://github.com/Lightning-AI/lightning/tree/master/src/lightning/pytorch>`__, we abstracted distributed training and hardware, by organizing PyTorch code.
With `Lightning Apps <https://github.com/Lightning-AI/lightning/tree/master/src/lightning/app>`__, we unified the local and cloud experience while abstracting infrastructure.

By using `PyTorch Lightning <https://github.com/Lightning-AI/lightning/tree/master/src/lightning/pytorch>`__ and `Lightning Apps <https://github.com/Lightning-AI/lightning/tree/master/src/lightning/app>`__
together, a completely new world of possibilities emerges.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/pl_to_app_4.png
    :alt: From PyTorch Lightning to Lightning App
    :width: 100 %

----

******************************************
1. Write an App to run the train.py script
******************************************

This article continues where the :ref:`build_model` guide finished.

Create an additional file ``app.py`` in the ``pl_project`` folder as follows:

.. code-block:: bash

    pl_project/
        app.py
        train.py
        requirements.txt

Inside the ``app.py`` file, add the following code.

.. literalinclude:: ../code_samples/convert_pl_to_app/app.py

This App runs the PyTorch Lightning script contained in the ``train.py`` file using the powerful :class:`~lightning.app.components.python.tracer.TracerPythonScript` component. This is really worth checking out!

----

************************************************
2. Run the train.py file locally or in the cloud
************************************************

First, go to the ``pl_folder`` folder from the local terminal and install the requirements.

.. code-block:: bash

    cd pl_folder
    pip install -r requirements.txt

To run your app, copy the following command to your local terminal:

.. code-block:: bash

    lightning_app run app app.py

Simply add ``--cloud`` to run this application in the cloud with a GPU machine 🤯

.. code-block:: bash

    lightning_app run app app.py --cloud


Congratulations! Now, you know how to run a `PyTorch Lightning <https://github.com/Lightning-AI/lightning/tree/master/src/lightning/pytorch>`_ script with Lightning Apps.

Lightning Apps can make your ML system way more powerful, keep reading to learn how.

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Level-up with Lightning Apps
   :description: From Basics to Advanced Skills
   :col_css: col-md-4
   :button_link: ../levels/basic/index.html
   :height: 180

.. displayitem::
   :header: Add an Interactive Demo
   :description: Add a Gradio Demo once the training is finished
   :col_css: col-md-4
   :button_link: add_an_interactive_demo.html
   :height: 180

.. displayitem::
   :header: Add Model Serving
   :description: Serve and load testing with MLServer and Locust
   :col_css: col-md-4
   :button_link: ../examples/model_server_app/model_server_app.html
   :height: 180

.. displayitem::
   :header: Add DAG Orchestration
   :description: Organize your processing, training and metrics collection
   :col_css: col-md-4
   :button_link: ../examples/dag/dag.html
   :height: 180

.. displayitem::
   :header: Add Team Collaboration
   :description: Create an app to run any PyTorch Lightning Script from Github
   :col_css: col-md-4
   :button_link: ../examples/github_repo_runner/github_repo_runner.html
   :height: 180
