************************************************
The *Train & Demo PyTorch Lightning* Application
************************************************

Find the *Train & Demo PyTorch Lightning* application in the `Lightning.ai App Gallery <https://lightning.ai/app/AU3WoWwdAP-Train%20%26%20Demo%20PyTorch%20Lightning>`_.

Here is a recording of this App running locally and in the cloud with the same behavior.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning_app_experience_cut.mp4
    :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning_app_experience_cut.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

In the steps below, we are going to show you how to build this application.

Here are `the entire App's code <https://github.com/Lightning-AI/lightning-quick-start>`_ and `its commented components. <https://github.com/Lightning-AI/lightning-quick-start/blob/main/quick_start/components.py>`_

----

*************************
Step 1: Install Lightning
*************************

If you are using a virtual env, don't forget to activate it before running commands.
You must do so in every new shell.

.. tip:: We highly recommend using virtual environments.

.. code:: bash

    pip install lightning

----

****************************************
Step 2: Install the *Train and Demo* App
****************************************
The first Lightning App we'll explore is an App to train and demo a machine learning model.

..
    [|qs_code|], [|qs_live_app|].

    .. |qs_live_app| raw:: html

       <a href="https://01g3w6gqdvjqjnqw05ccy69nwy.litng-ai-03.litng.ai/view/Interactive%20demo" target="_blank">live app</a>

    .. |qs_code| raw:: html

       <a href="https://github.com/Lightning-AI/lightning-quick-start" target="_blank">code</a>


Install this App by typing:

.. code-block:: bash

    lightning_app install app lightning/quick-start

Verify the App was successfully installed:

.. code-block:: bash

    cd lightning-quick-start

----

***************************
Step 3: Run the App locally
***************************

Run the app locally with the ``run`` command 🤯

.. code:: bash

    lightning_app run app app.py

----

********************************
Step 4: Run the App in the cloud
********************************

Add the ``--cloud`` argument to run on the `Lightning.AI cloud <http://lightning.ai/>`_. 🤯🤯🤯

.. code:: bash

    lightning_app run app app.py --cloud

..
    Your app should look like this one (|qs_live_app|)

----

*******************
Understand the code
*******************
The App that we just launched trained a PyTorch Lightning model (although any framework works), then added an interactive demo.

This is the App's code:

.. code:: python

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute())

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

Let's break down the code section by section to understand what it is doing.

----

1: Define root component
^^^^^^^^^^^^^^^^^^^^^^^^

A Lightning App provides a cohesive product experience for a set of unrelated components.

The top-level component (Root) must subclass ``L.LightningFlow``


.. code:: python
    :emphasize-lines: 6

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu-small"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

----

2: Define components
^^^^^^^^^^^^^^^^^^^^
In the __init__ method, we define the components that make up the App. In this case, we have 2 components,
a component to execute any PyTorch Lightning script (model training) and a second component to
start a Gradio server for demo purposes.

.. code:: python
    :emphasize-lines: 9, 14

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu-small"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

----

3: Define how components Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Every component has a ``run`` method. The run method defines the 🌊 Flow 🌊 of how components interact together.

In this case, we train a model (until completion). When it's done AND there exists a checkpoint, we launch a
demo server:

.. code:: python
    :emphasize-lines: 18, 21, 22

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu-small"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

.. note:: If you've used other ML systems you'll be pleasantly surprised to not find decorators or YAML files.

----

4: Connect web user interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All our favorite tools normally have their own web user interfaces (UI).

Implement the ``configure_layout`` method to connect them together:

.. code:: python
    :emphasize-lines: 24-27

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu-small"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

----

5: Init the ``app`` object
^^^^^^^^^^^^^^^^^^^^^^^^^^
Initialize an ``app`` object with the ``TrainDeploy`` component (this won't run the App yet):

.. code:: python
    :emphasize-lines: 29

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio

    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu-small"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]

    app = L.LightningApp(TrainDeploy())

----

******************************
What components are supported?
******************************
Any component can work with Lightning AI!

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/Lightning.gif
    :alt: What is Lightning gif.
    :width: 100 %

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add components to your App
   :description: Expand your App by adding components.
   :col_css: col-md-4
   :button_link: ../workflows/extend_app.html
   :height: 180

.. displayitem::
   :header: Build a component
   :description: Learn to build your own component.
   :col_css: col-md-4
   :button_link: ../workflows/build_lightning_component/index.html
   :height: 180

.. displayitem::
   :header: Explore more Apps
   :description: Explore more apps for inspiration.
   :col_css: col-md-4
   :button_link: https://lightning.ai/apps
   :height: 180

.. displayitem::
   :header: Under the hood
   :description: Explore how it works under the hood.
   :col_css: col-md-4
   :button_link: ../core_api/lightning_app/index.html
   :height: 180

.. displayitem::
   :header: Run on your private cloud
   :description: Run Lightning Apps on your private VPC or on-prem.
   :button_link: ../workflows/run_on_private_cloud.html
   :col_css: col-md-4
   :height: 180

.. raw:: html

        </div>
    </div>
