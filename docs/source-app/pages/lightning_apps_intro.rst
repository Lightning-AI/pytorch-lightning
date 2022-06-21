.. toctree::
    :maxdepth: 1
    :hidden:

    ../workflows/extend_app
    ../workflows/build_lightning_component/index
    ../glossary/lightning_app_overview/index
    ../workflows/run_on_private_cloud


############################
Lightning Apps in 15 minutes
############################

**Required background:** Basic Python familiarity.

**Goal:** In this guide, we'll walk you through the 4 key steps to build your first Lightning app.

----

The app we build in this guide trains and deploys a model.

..
    (|qs_app|).

    .. |qs_app| raw:: html

       <a href="https://01g3ptaq9ccd3ksz8r3yzmjpcs.litng-ai-03.litng.ai/view/Interactive%20demo" target="_blank">see the app live here</a>


A Lightning app is **Organized Python**, it enables AI researchers and ML engineers to build complex AI workflows without any of the **cloud** boilerplate.

With Lightning apps your favorite components can work together on any machine at any scale. Here's an illustration:

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/apps_vid.mp4"></video>

|

Lightning Apps are:

- cloud agnostic
- fault-tolerant
- production ready
- locally debuggable
- and much more

.. join_slack::
   :align: left
   :margin: 20


----

*******************
Who can build apps?
*******************
Anyone who knows Python can build a Lightning app, even without ML experience.

----

*************************
Step 1: Install Lightning
*************************

First, you'll need to install Lightning (make sure you use Python 3.8+).

.. code:: bash

    python -m pip install -U lightning

(pip and conda install coming soon)

----

********************************
Step 2: Install Train Deploy App
********************************
The first Lightning app we'll explore is an app to train and deploy a machine learning model.

..
    [|qs_code|], [|qs_live_app|].

    .. |qs_live_app| raw:: html

       <a href="https://01g3w6gqdvjqjnqw05ccy69nwy.litng-ai-03.litng.ai/view/Interactive%20demo" target="_blank">live app</a>

    .. |qs_code| raw:: html

       <a href="https://github.com/PyTorchLightning/lightning-quick-start" target="_blank">code</a>


Install this app by typing:

.. code-block:: bash

    lightning install app lightning/quick-start

Verify the app was succesfully installed:

.. code-block:: bash

    cd lightning-quick-start
    ls

----

***************************
Step 3: Run the app locally
***************************
Run the app locally with the ``run`` command

.. code:: bash

    lightning run app app.py

🤯

----

********************************
Step 4: Run the app on the cloud
********************************
Add the ``--cloud`` argument to run on the `Lightning.AI cloud <http://lightning.ai/>`_.

.. code:: bash

    lightning run app app.py --cloud

🤯🤯🤯

..
    Your app should look like this one (|qs_live_app|)


----

*******************
Understand the code
*******************
The app that we just launched trained a PyTorch Lightning model (although any framework works), then added an interactive demo.

This is the app code:

.. code:: python

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute())

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


    app = lapp.LightningApp(TrainDeploy())

Let's break down the app code by each section to understand what it is doing.

----

1: Define root component
^^^^^^^^^^^^^^^^^^^^^^^^
A Lightning app provides a cohesive product experience for a set of unrelated components.

The top-level component (Root) must subclass lapp.LightningFlow


.. code:: python
    :emphasize-lines: 6

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute("cpu-small"))

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


    app = lapp.LightningApp(TrainDeploy())

----

2: Define components
^^^^^^^^^^^^^^^^^^^^
In the __init__ method, we define the components that make up the app. In this case, we have 2 components,
a component to execute any pytorch lightning script (model training) and a second component to
start a gradio server for demo purposes.

.. code:: python
    :emphasize-lines: 9, 14

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute("cpu-small"))

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


    app = lapp.LightningApp(TrainDeploy())

----

3: Define how components flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Every component has a `run` method. The run method defines the 🌊 flow 🌊 of how components interact together.

In this case, we train a model (until completion). When it's done AND there exists a checkpoint, we launch a
demo server:

.. code:: python
    :emphasize-lines: 18, 21, 22

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute("cpu-small"))

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


    app = lapp.LightningApp(TrainDeploy())

..
  If you've used other ML systems you'll be pleasantly surprised to not find decorators or YAML files.
  Read here to understand the benefits more.

----

4: Connect web user interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All our favorite tools normally have their own web user interfaces (UI).

Implement the `configure_layout` method to connect them together:

.. code:: python
    :emphasize-lines: 24-27

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute("cpu-small"))

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


    app = lapp.LightningApp(TrainDeploy())

----

5: Init the app object
^^^^^^^^^^^^^^^^^^^^^^
Initalize an app object with the `TrainDeploy` component (this won't run the app yet):

.. code:: python
    :emphasize-lines: 29

    # lightning-quick-start/app.py
    import os.path as ops
    import lightning_app as la
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(lapp.CloudCompute("cpu-small"))

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


    app = lapp.LightningApp(TrainDeploy())

----

******************************
What components are supported?
******************************
Any component can work with Lightning AI!

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/Lightning.gif
    :alt: What is Lightning gif.
    :width: 100 %

----

**********
Next steps
**********
Depending on your use case, you might want to check one of these out next.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add components to your app
   :description: Expand your app by adding components.
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
   :header: Explore more apps
   :description: Explore more apps for inspiration.
   :col_css: col-md-4
   :button_link: https://lightning.ai/apps
   :height: 180

.. displayitem::
   :header: How it works under the hood
   :description: Explore how it works under the hood.
   :col_css: col-md-4
   :button_link: core_api/lightning_app/index.html
   :height: 180

.. displayitem::workflo
   :header: Run on your private cloud
   :description: Run lightning apps on your private VPC or on-prem.
   :button_link: ../workflows/run_on_private_cloud.html
   :col_css: col-md-4
   :height: 180

.. raw:: html

        </div>
    </div>
